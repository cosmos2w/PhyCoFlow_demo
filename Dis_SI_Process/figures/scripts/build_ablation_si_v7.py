#!/usr/bin/env python
"""Build the Figure 5 V7 supplementary ablation package.

This renderer is deliberately post-processing only.  It consumes the saved
evaluation CSV/JSON artifacts in the frozen evaluation directory and writes
only SI artwork plus a machine-readable plot manifest.  The six SI entry
figures are accompanied by every meaningful subpanel as editable SVG and
600-dpi PNG; no PDF is produced by this script.

The renderer can read the compact source files produced by the V7 collector
when present, but the evaluation artifacts remain the authoritative source.
The direct evaluation fallback is useful when the collector and renderer are
run independently and still keeps every plotted row traceable to its raw
source path and hash.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import LogLocator, NullFormatter


# Required by the Nature-figure workflow before any figure is created.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.hashsalt"] = "figure5-v7-ablation"


SCRIPT_PATH = Path(__file__).resolve()
# ``.../Dis_SI_Process/figures/scripts/build_ablation_si_v7.py``:
# parents[0]=scripts, [1]=figures, [2]=Dis_SI_Process, [3]=repository.
# Keep the repository root separate from the package root so evaluation paths
# and output paths resolve to the workspace rather than under Dis_SI_Process/.
PACKAGE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "figure5_v7_ablation.yaml"
DEFAULT_EVAL = (
    REPO_ROOT
    / "0_demo_TurbulentCombustion"
    / "Save_TrainedModel"
    / "ablation_condT"
    / "evaluation_20260910"
)

STOCHASTIC_ORDER = ["A0", "A2", "A3", "A5", "A4"]
ALL_METHODS = ["A0", "A1", "A2", "A3", "A4", "A5"]
FIELDS = ["CH4", "CO", "T", "U1", "p"]
UNOBSERVED_FIELDS = ["CH4", "CO", "U1", "p"]
PAIR_TARGETS = ["T-U1", "CH4-U1", "p-U1"]
HISTORY_COLUMNS = ("epoch", "train_loss", "val_loss")

DEFAULT_STYLES: dict[str, dict[str, Any]] = {
    "A0": {"label": "Full model", "color": "#E63946", "marker": "o"},
    "A2": {"label": "No sensor feedback", "color": "#64748B", "marker": "s"},
    "A3": {"label": "No local conditioning", "color": "#7D8794", "marker": "^"},
    "A5": {"label": "Local-only conditioning", "color": "#485564", "marker": "v"},
    "A4": {"label": "IID Gaussian prior", "color": "#BE873D", "marker": "D"},
    "A1": {"label": "Deterministic regression", "color": "#596674", "marker": "s"},
}


def _hex_with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    rgb = mpl.colors.to_rgb(color)
    return (*rgb, alpha)


def apply_style(config: Mapping[str, Any]) -> None:
    fig_cfg = config.get("figure", {})
    mpl.rcParams.update(
        {
            "font.size": float(fig_cfg.get("row_tick_pt", 6.0)),
            "axes.labelsize": float(fig_cfg.get("axis_label_pt", 6.6)),
            "xtick.labelsize": float(fig_cfg.get("row_tick_pt", 6.0)),
            "ytick.labelsize": float(fig_cfg.get("row_tick_pt", 6.0)),
            "legend.fontsize": float(fig_cfg.get("row_tick_pt", 6.0)),
            "axes.titlesize": float(fig_cfg.get("axis_label_pt", 6.6)),
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_config(path: Path) -> dict[str, Any]:
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        loaded = {}
    styles = dict(DEFAULT_STYLES)
    styles.update(loaded.get("styles", {}))
    loaded["styles"] = styles
    loaded.setdefault("stochastic_order", STOCHASTIC_ORDER)
    return loaded


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite plotted value: {value!r}")
    return result


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def style_for(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    result = dict(DEFAULT_STYLES[method])
    result.update(config.get("styles", {}).get(method, {}))
    return result


def method_label(config: Mapping[str, Any], method: str) -> str:
    return str(style_for(config, method).get("label", DEFAULT_STYLES[method]["label"]))


@dataclass
class SourceBundle:
    evaluation_root: Path
    last_summary: pd.DataFrame
    best_summary: pd.DataFrame
    last_per_state: pd.DataFrame
    best_per_state: pd.DataFrame
    last_paired: pd.DataFrame
    best_paired: pd.DataFrame
    hf_summary: pd.DataFrame
    hf_paired: pd.DataFrame
    hf_per_state: pd.DataFrame
    population_spectra: pd.DataFrame
    hann_summary: pd.DataFrame
    sensitivity: pd.DataFrame
    provenance: dict[str, dict[str, Any]]
    histories: dict[str, pd.DataFrame]
    source_paths: dict[str, Path]
    source_hashes: dict[str, str]


@dataclass
class PlotManifest:
    timestamp: str
    renderer: str
    config_path: str
    evaluation_root: str
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    plots: list[dict[str, Any]] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)
    qa: dict[str, Any] = field(default_factory=dict)

    def add_source(self, key: str, path: Path, *, role: str, rows: int | None = None) -> None:
        self.sources[key] = {
            "path": relpath(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "role": role,
            "rows": rows,
        }

    def add_plot(self, record: dict[str, Any]) -> None:
        self.plots.append(record)

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "figure5-v7-ablation-si-plot-manifest-1",
                    "timestamp": self.timestamp,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "renderer": self.renderer,
                    "config_path": self.config_path,
                    "evaluation_root": self.evaluation_root,
                    "scope": self.scope,
                    "sources": self.sources,
                    "plots": self.plots,
                    "qa": self.qa,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )


def _history_candidates(eval_root: Path, method: str, provenance: Mapping[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    run_dir = provenance.get("run_directory")
    if run_dir:
        candidates.append(Path(str(run_dir)) / "loss_history.csv")
    # The run directory is also recoverable from the frozen checkpoint path.
    for checkpoint_key in ("checkpoint_path", "checkpoints"):
        value = provenance.get(checkpoint_key)
        if isinstance(value, str):
            candidates.append(Path(value).parent / "loss_history.csv")
        elif isinstance(value, Mapping):
            for item in value.values():
                if isinstance(item, Mapping) and item.get("path"):
                    candidates.append(Path(str(item["path"])).parent / "loss_history.csv")
    # A0's frozen checkpoint lives below ``evaluation_20260910`` and therefore
    # cannot identify its source history from ``checkpoint_path``.  Resolve it
    # through the frozen snapshot manifest's explicit source run directory.
    if method == "A0":
        snapshot = eval_root / "checkpoint_inputs" / "A0" / "snapshot_manifest.json"
        if snapshot.is_file():
            try:
                source_run = json.loads(snapshot.read_text(encoding="utf-8")).get("source_run_directory")
                if source_run:
                    candidates.append(Path(str(source_run)) / "loss_history.csv")
            except (OSError, json.JSONDecodeError):
                pass
    # For non-frozen variants, the checkpoint provenance points directly to
    # the run directory.  Do not glob by A-code: multiple historical runs can
    # coexist and selecting the first one would break checkpoint lineage.
    return list(dict.fromkeys(candidates))


def _history_snapshot_paths(snapshot_dir: Path) -> dict[str, Path]:
    return {method: snapshot_dir / f"training_history_{method}.csv" for method in STOCHASTIC_ORDER}


def _stable_history_bytes(raw_paths: Mapping[str, Path], *, attempts: int = 5) -> tuple[dict[str, bytes], dict[str, str], dict[str, int]]:
    """Read each live history as a stable byte snapshot.

    A running training job can append to ``loss_history.csv`` while the SI
    renderer is reading it.  Hashing before and after the byte read makes the
    capture fail closed rather than silently mixing rows from two revisions.
    """
    for _attempt in range(attempts):
        before = {method: sha256_file(path) for method, path in raw_paths.items()}
        mtimes_before = {method: path.stat().st_mtime_ns for method, path in raw_paths.items()}
        payloads = {method: path.read_bytes() for method, path in raw_paths.items()}
        payload_hashes = {method: hashlib.sha256(payload).hexdigest() for method, payload in payloads.items()}
        after = {method: sha256_file(path) for method, path in raw_paths.items()}
        mtimes_after = {method: path.stat().st_mtime_ns for method, path in raw_paths.items()}
        if all(before[m] == payload_hashes[m] == after[m] and mtimes_before[m] == mtimes_after[m] for m in raw_paths):
            return payloads, payload_hashes, mtimes_after
    raise RuntimeError("training history changed while capturing the timestamped SI snapshot")


def capture_history_snapshot(eval_root: Path, snapshot_dir: Path, timestamp: str) -> dict[str, Any]:
    """Freeze all recorded rows for the five stochastic SI history curves.

    The compact CSVs are timestamp-local SI inputs.  Once their manifest is
    present, reruns validate and reuse those bytes rather than reopening a
    live training history, which preserves SVG reproducibility.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = snapshot_dir / "history_snapshot_manifest.json"
    snapshot_paths = _history_snapshot_paths(snapshot_dir)
    if manifest_path.is_file():
        snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
        if snapshot.get("schema_version") != "figure5-v7-ablation-history-snapshot-1":
            raise RuntimeError(f"unsupported history snapshot schema in {manifest_path}")
        if snapshot.get("timestamp") != timestamp:
            raise RuntimeError(f"history snapshot timestamp mismatch in {manifest_path}")
        for method, path in snapshot_paths.items():
            info = snapshot.get("histories", {}).get(method, {})
            if not path.is_file():
                raise FileNotFoundError(f"history snapshot is incomplete: {path}")
            if sha256_file(path) != info.get("snapshot_sha256"):
                raise RuntimeError(f"history snapshot hash mismatch: {path}")
            frame = read_csv(path)
            if tuple(frame.columns) != HISTORY_COLUMNS:
                raise RuntimeError(f"history snapshot columns changed: {path}")
            if len(frame) != int(info.get("captured_rows", -1)):
                raise RuntimeError(f"history snapshot row count changed: {path}")
        return snapshot

    raw_paths: dict[str, Path] = {}
    for method in STOCHASTIC_ORDER:
        provenance_path = eval_root / "last" / f"provenance_{method}.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
        raw_path = next((p for p in _history_candidates(eval_root, method, provenance) if p.is_file()), None)
        if raw_path is None:
            raise FileNotFoundError(f"no recorded training history found for {method}")
        raw_paths[method] = raw_path.resolve()

    payloads, raw_hashes, raw_mtimes = _stable_history_bytes(raw_paths)
    captured_at = datetime.now(timezone.utc).isoformat()
    staging = snapshot_dir / f".history_snapshot_staging_{os.getpid()}"
    if staging.exists():
        for path in staging.iterdir():
            path.unlink()
    else:
        staging.mkdir(parents=True)
    records: dict[str, dict[str, Any]] = {}
    try:
        for method in STOCHASTIC_ORDER:
            raw_path = raw_paths[method]
            frame = pd.read_csv(io.BytesIO(payloads[method]))
            missing = [column for column in HISTORY_COLUMNS if column not in frame.columns]
            if missing:
                raise ValueError(f"history {method} is missing required columns: {missing}")
            compact = frame.loc[:, list(HISTORY_COLUMNS)].copy()
            staged_path = staging / f"training_history_{method}.csv"
            compact.to_csv(staged_path, index=False, na_rep="")
            snapshot_hash = sha256_file(staged_path)
            records[method] = {
                "raw_source_path": str(raw_path),
                "raw_source_relpath": relpath(raw_path),
                "raw_sha256_at_capture": raw_hashes[method],
                "raw_size_bytes": len(payloads[method]),
                "raw_mtime_ns_at_capture": raw_mtimes[method],
                "captured_rows": int(len(compact)),
                "captured_columns": list(HISTORY_COLUMNS),
                "snapshot_path": relpath(snapshot_paths[method]),
                "snapshot_sha256": snapshot_hash,
                "source_unchanged_during_capture": True,
            }
        snapshot = {
            "schema_version": "figure5-v7-ablation-history-snapshot-1",
            "timestamp": timestamp,
            "captured_at_utc": captured_at,
            "evaluation_root": relpath(eval_root),
            "recorded_scope": "all rows present in each source loss_history.csv at capture; no truncation, interpolation, smoothing, or endpoint reconstruction",
            "methods": list(STOCHASTIC_ORDER),
            "histories": records,
        }
        staged_manifest = staging / manifest_path.name
        staged_manifest.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for method in STOCHASTIC_ORDER:
            os.replace(staging / snapshot_paths[method].name, snapshot_paths[method])
        os.replace(staged_manifest, manifest_path)
    finally:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
    return snapshot


def load_sources(eval_root: Path, manifest: PlotManifest, *, history_snapshot_dir: Path | None = None) -> SourceBundle:
    metrics_paths: dict[str, Path] = {}
    for policy in ("last", "best"):
        policy_dir = eval_root / policy / "metrics"
        for name in ("summary_metrics.csv", "per_state_metrics.csv", "paired_differences.csv"):
            path = policy_dir / name
            metrics_paths[f"{policy}_{name.removesuffix('.csv')}"] = path
    hf_dir = eval_root / "high_frequency"
    metrics_paths.update(
        {
            "high_frequency_summary": hf_dir / "summary_high_frequency.csv",
            "high_frequency_paired": hf_dir / "paired_high_frequency.csv",
            "high_frequency_per_state": hf_dir / "per_state_high_frequency.csv",
            "population_U1_spectra": hf_dir / "population_U1_spectra.csv",
            "U1_hann_robustness_summary": hf_dir / "U1_hann_robustness_summary.csv",
            "checkpoint_sensitivity": eval_root / "checkpoint_sensitivity_vs_last.csv",
        }
    )
    for key, path in metrics_paths.items():
        manifest.add_source(key, path, role="saved evaluation source")
    extra_json = [
        eval_root / "evaluation_environment.json",
        eval_root / "validation_summary.json",
        eval_root / "checkpoint_inputs" / "A0" / "snapshot_manifest.json",
        eval_root / "source_hashes.json",
        eval_root / "A0_training_progress.csv",
    ]
    for path in extra_json:
        manifest.add_source(path.stem, path, role="evaluation provenance")

    required = [path for path in metrics_paths.values() if not path.is_file()]
    if required:
        missing = "\n".join(relpath(path) for path in required)
        raise FileNotFoundError(f"missing required SI source files:\n{missing}")

    last_summary = read_csv(metrics_paths["last_summary_metrics"])
    best_summary = read_csv(metrics_paths["best_summary_metrics"])
    last_per_state = read_csv(metrics_paths["last_per_state_metrics"])
    best_per_state = read_csv(metrics_paths["best_per_state_metrics"])
    last_paired = read_csv(metrics_paths["last_paired_differences"])
    best_paired = read_csv(metrics_paths["best_paired_differences"])
    hf_summary = read_csv(metrics_paths["high_frequency_summary"])
    hf_paired = read_csv(metrics_paths["high_frequency_paired"])
    hf_per_state = read_csv(metrics_paths["high_frequency_per_state"])
    population_spectra = read_csv(metrics_paths["population_U1_spectra"])
    hann_summary = read_csv(metrics_paths["U1_hann_robustness_summary"])
    sensitivity = read_csv(metrics_paths["checkpoint_sensitivity"])
    dataframes = {
        "last_summary": last_summary,
        "best_summary": best_summary,
        "last_per_state": last_per_state,
        "best_per_state": best_per_state,
        "last_paired": last_paired,
        "best_paired": best_paired,
        "high_frequency_summary": hf_summary,
        "high_frequency_paired": hf_paired,
        "high_frequency_per_state": hf_per_state,
        "population_U1_spectra": population_spectra,
        "U1_hann_robustness_summary": hann_summary,
        "checkpoint_sensitivity": sensitivity,
    }
    for key, df in dataframes.items():
        source_key = {
            "last_summary": "last_summary_metrics",
            "best_summary": "best_summary_metrics",
            "last_per_state": "last_per_state_metrics",
            "best_per_state": "best_per_state_metrics",
            "last_paired": "last_paired_differences",
            "best_paired": "best_paired_differences",
            "high_frequency_summary": "high_frequency_summary",
            "high_frequency_paired": "high_frequency_paired",
            "high_frequency_per_state": "high_frequency_per_state",
            "population_U1_spectra": "population_U1_spectra",
            "U1_hann_robustness_summary": "U1_hann_robustness_summary",
            "checkpoint_sensitivity": "checkpoint_sensitivity",
        }[key]
        manifest.sources[source_key]["rows"] = int(len(df))

    provenance: dict[str, dict[str, Any]] = {}
    for policy in ("last", "best"):
        for method in ALL_METHODS:
            path = eval_root / policy / f"provenance_{method}.json"
            manifest.add_source(f"{policy}_provenance_{method}", path, role="checkpoint provenance")
            if path.is_file():
                provenance[f"{policy}_{method}"] = json.loads(path.read_text(encoding="utf-8"))
            manifest_path = eval_root / policy / f"manifest_{method}.csv"
            manifest.add_source(f"{policy}_manifest_{method}", manifest_path, role="frozen evaluation cache manifest")

    histories: dict[str, pd.DataFrame] = {}
    if history_snapshot_dir is not None:
        snapshot_manifest_path = history_snapshot_dir / "history_snapshot_manifest.json"
        if not snapshot_manifest_path.is_file():
            raise FileNotFoundError(f"missing timestamped history snapshot: {snapshot_manifest_path}")
        snapshot = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        manifest.add_source("history_snapshot_manifest", snapshot_manifest_path, role="timestamp-frozen training-history provenance", rows=len(snapshot.get("histories", {})))
        snapshot_paths = _history_snapshot_paths(history_snapshot_dir)
        for method in STOCHASTIC_ORDER:
            history_path = snapshot_paths[method]
            histories[method] = read_csv(history_path)
            if tuple(histories[method].columns) != HISTORY_COLUMNS:
                raise ValueError(f"timestamped history snapshot has unexpected columns: {history_path}")
            info = snapshot.get("histories", {}).get(method, {})
            source_info = {
                "raw_source_path": info.get("raw_source_path"),
                "raw_source_relpath": info.get("raw_source_relpath"),
                "raw_sha256_at_capture": info.get("raw_sha256_at_capture"),
                "captured_rows": info.get("captured_rows"),
                "snapshot_sha256": info.get("snapshot_sha256"),
                "captured_at_utc": snapshot.get("captured_at_utc"),
            }
            manifest.add_source(f"history_{method}", history_path, role="timestamp-frozen compact recorded training history", rows=len(histories[method]))
            manifest.sources[f"history_{method}"].update(source_info)
    else:
        # Direct function callers retain the raw-source fallback, while the
        # CLI always supplies a timestamped snapshot directory.
        for method in STOCHASTIC_ORDER:
            candidate_prov = provenance.get(f"last_{method}", {})
            history_path = next((p for p in _history_candidates(eval_root, method, candidate_prov) if p.is_file()), None)
            if history_path is not None:
                histories[method] = read_csv(history_path)
                manifest.add_source(f"history_{method}", history_path, role="recorded training history", rows=len(histories[method]))

    hashes = {key: info["sha256"] for key, info in manifest.sources.items() if info.get("sha256")}
    return SourceBundle(
        evaluation_root=eval_root,
        last_summary=last_summary,
        best_summary=best_summary,
        last_per_state=last_per_state,
        best_per_state=best_per_state,
        last_paired=last_paired,
        best_paired=best_paired,
        hf_summary=hf_summary,
        hf_paired=hf_paired,
        hf_per_state=hf_per_state,
        population_spectra=population_spectra,
        hann_summary=hann_summary,
        sensitivity=sensitivity,
        provenance=provenance,
        histories=histories,
        source_paths={key: path for key, path in metrics_paths.items()},
        source_hashes=hashes,
    )


def validate_sources(data: SourceBundle, strict: bool) -> dict[str, Any]:
    expected_methods = set(ALL_METHODS)
    checks: dict[str, Any] = {}
    checks["expected_methods"] = sorted(expected_methods)
    checks["summary_methods_last"] = sorted(set(data.last_summary["method"].astype(str)))
    checks["summary_methods_best"] = sorted(set(data.best_summary["method"].astype(str)))
    checks["per_state_unique_keys"] = {}
    errors: list[str] = []
    for policy, frame in (("last", data.last_per_state), ("best", data.best_per_state)):
        if not {"method", "snapshot", "time_index", "metric", "target", "value"}.issubset(frame.columns):
            errors.append(f"{policy} per-state schema is incomplete")
            continue
        subset = frame[frame["method"].isin(ALL_METHODS)]
        counts = subset.groupby(["method", "metric", "target"], dropna=False).size()
        checks["per_state_unique_keys"][policy] = {
            "methods": sorted(set(subset["method"])),
            # A row exists for each metric/target at a state.  The complete
            # identity therefore includes metric and target; checking only
            # method/snapshot/time incorrectly flags the intentionally long
            # format source as duplicated.
            "duplicate_method_snapshot_time_metric_target": int(
                subset.duplicated(["method", "snapshot", "time_index", "metric", "target"]).sum()
            ),
            "metric_target_min_count": int(counts.min()) if len(counts) else 0,
            "metric_target_max_count": int(counts.max()) if len(counts) else 0,
        }
        if strict:
            if set(subset["method"]) != expected_methods:
                errors.append(f"{policy} per-state methods incomplete")
            if checks["per_state_unique_keys"][policy]["duplicate_method_snapshot_time_metric_target"]:
                errors.append(f"{policy} per-state duplicate method/snapshot/time/metric/target keys")
            expected = subset.groupby(["method", "metric", "target"], dropna=False).size()
            if not (expected == 1000).all():
                errors.append(f"{policy} per-state rows are not exactly 1000 per metric/target")
        if not np.isfinite(pd.to_numeric(subset["value"], errors="coerce")).all():
            errors.append(f"{policy} per-state contains non-finite values")

    if strict:
        for policy, frame in (("last", data.last_summary), ("best", data.best_summary)):
            if set(frame["method"].astype(str)) != expected_methods:
                errors.append(f"{policy} summary methods incomplete")
            if not (pd.to_numeric(frame["n"], errors="coerce") == 1000).all():
                errors.append(f"{policy} summary contains non-1000 n")
        if set(data.hf_summary["method"].astype(str)) != expected_methods:
            errors.append("high-frequency summary methods incomplete")
        if set(data.hf_summary["policy"].astype(str)) != {"last", "best"}:
            errors.append("high-frequency summary policies incomplete")
        if not (pd.to_numeric(data.hf_summary["n"], errors="coerce") == 1000).all():
            errors.append("high-frequency summary contains non-1000 n")
        if not {"A0", "A4"}.issubset(set(data.population_spectra["method"].astype(str))):
            errors.append("population spectra is missing the full or IID prior rows")
        if set(data.population_spectra["policy"].astype(str)) != {"last", "best"}:
            errors.append("population spectra policies incomplete")
        expected_paths = [
            data.evaluation_root / "evaluation_environment.json",
            data.evaluation_root / "validation_summary.json",
            data.evaluation_root / "checkpoint_inputs" / "A0" / "snapshot_manifest.json",
        ]
        for path in expected_paths:
            if not path.is_file():
                errors.append(f"missing provenance file: {relpath(path)}")
        # Keep the two essential direction checks visible in QA.
        macro = {
            policy: {
                method: summary_value(
                    data,
                    policy,
                    method,
                    "physical_relative_l2",
                    "Unobserved_mean",
                )["mean"]
                for method in STOCHASTIC_ORDER
            }
            for policy in ("last", "best")
        }
        checks["macro_direction"] = macro
        for policy in ("last", "best"):
            if not macro[policy]["A4"] < macro[policy]["A0"]:
                errors.append(f"IID is not lower macro L2 under {policy}")
            for field_name in FIELDS:
                full = hf_value(data, policy, "A0", field_name, "highband_error_relative_l2")["mean"]
                iid = hf_value(data, policy, "A4", field_name, "highband_error_relative_l2")["mean"]
                if not iid > full:
                    errors.append(f"IID high-band residual is not higher for {field_name} under {policy}")
    checks["status"] = "pass" if not errors else "fail"
    checks["errors"] = errors
    if strict and errors:
        raise RuntimeError("strict SI source validation failed:\n" + "\n".join(errors))
    return checks


def summary_frame(data: SourceBundle, policy: str) -> pd.DataFrame:
    return data.last_summary if policy == "last" else data.best_summary


def paired_frame(data: SourceBundle, policy: str) -> pd.DataFrame:
    return data.last_paired if policy == "last" else data.best_paired


def summary_value(
    data: SourceBundle,
    policy: str,
    method: str,
    metric: str,
    target: str,
) -> dict[str, Any]:
    frame = summary_frame(data, policy)
    rows = frame[(frame["method"] == method) & (frame["metric"] == metric) & (frame["target"] == target)]
    if len(rows) != 1:
        raise KeyError(f"expected one summary row for {policy}/{method}/{metric}/{target}, got {len(rows)}")
    row = rows.iloc[0]
    return {
        "mean": safe_float(row["mean"]),
        "low": safe_float(row["block20_ci95_low"]),
        "high": safe_float(row["block20_ci95_high"]),
        "n": int(row["n"]),
        "q25": safe_float(row["q25"]) if "q25" in row else None,
        "q75": safe_float(row["q75"]) if "q75" in row else None,
        "fraction_gt_1": safe_float(row["fraction_gt_1"]) if "fraction_gt_1" in row else None,
        "fraction_gt_2": safe_float(row["fraction_gt_2"]) if "fraction_gt_2" in row else None,
    }


def hf_value(data: SourceBundle, policy: str, method: str, field_name: str, metric: str) -> dict[str, Any]:
    rows = data.hf_summary[
        (data.hf_summary["policy"] == policy)
        & (data.hf_summary["method"] == method)
        & (data.hf_summary["field"] == field_name)
        & (data.hf_summary["metric"] == metric)
    ]
    if len(rows) != 1:
        raise KeyError(f"expected one high-frequency row for {policy}/{method}/{field_name}/{metric}, got {len(rows)}")
    row = rows.iloc[0]
    return {
        "mean": safe_float(row["mean"]),
        "low": safe_float(row["block20_ci95_low"]),
        "high": safe_float(row["block20_ci95_high"]),
        "n": int(row["n"]),
        "q25": safe_float(row["q25"]) if "q25" in row else None,
        "q75": safe_float(row["q75"]) if "q75" in row else None,
        "fraction_gt_1": safe_float(row["fraction_gt_1"]) if "fraction_gt_1" in row else None,
        "fraction_gt_2": safe_float(row["fraction_gt_2"]) if "fraction_gt_2" in row else None,
    }


def paired_value(
    data: SourceBundle,
    policy: str,
    method: str,
    baseline: str,
    metric: str,
    target: str,
    *,
    high_frequency: bool = False,
) -> dict[str, Any]:
    frame = data.hf_paired if high_frequency else paired_frame(data, policy)
    if high_frequency:
        rows = frame[
            (frame["policy"] == policy)
            & (frame["method"] == method)
            & (frame["baseline"] == baseline)
            & (frame["field"] == target)
            & (frame["metric"] == metric)
        ]
    else:
        rows = frame[
            (frame["method"] == method)
            & (frame["baseline"] == baseline)
            & (frame["metric"] == metric)
            & (frame["target"] == target)
        ]
    if len(rows) != 1:
        raise KeyError(f"expected one paired row for {policy}/{method}/{baseline}/{metric}/{target}, got {len(rows)}")
    row = rows.iloc[0]
    return {
        "difference": safe_float(row["method_mean_minus_baseline"] if high_frequency else row["mean_difference"]),
        "relative_percent": safe_float(row["relative_mean_change_percent"]),
        "low": safe_float(row["block20_ci95_low"]),
        "high": safe_float(row["block20_ci95_high"]),
        "low5": safe_float(row["block5_ci95_low"]),
        "high5": safe_float(row["block5_ci95_high"]),
        "low50": safe_float(row["block50_ci95_low"]),
        "high50": safe_float(row["block50_ci95_high"]),
        "fraction_less": safe_float(row["fraction_method_less_than_baseline"]),
        "n": int(row["n_paired"]),
    }


def add_panel_tag(ax: Any, label: str, *, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, fontweight="bold")


def clean_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.2)


def _adopt_axes(
    axes: Any,
    figure: Any,
    *,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
    sharex: Any = False,
    sharey: Any = False,
) -> tuple[Any, np.ndarray, bool]:
    """Return an axes array for standalone or nested SI rendering.

    Entry-point SI composites use nested ``SubFigure`` objects so that the
    detailed standalone plotting functions can be reused without raster
    imports or compact redraws that silently drop policies/intervals.  The
    boolean records ownership: only a standalone-owned figure should call
    ``tight_layout`` and be closed by the caller.
    """
    if axes is None:
        fig, arr = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
        arr = np.asarray(arr, dtype=object).reshape(nrows, ncols)
        return fig, (arr.ravel() if nrows == 1 or ncols == 1 else arr), True
    fig = figure if figure is not None else np.asarray(axes, dtype=object).flat[0].figure
    arr = np.asarray(axes, dtype=object).reshape(nrows, ncols)
    return fig, (arr.ravel() if nrows == 1 or ncols == 1 else arr), False


def _visual_qa(fig: Any, *, crop: bool = True) -> dict[str, Any]:
    """Check rendered text extents against the tight export box.

    This catches labels/legends that survive SVG parsing but are clipped at
    final-size export.  Interval clipping is checked separately from the
    machine-readable plot coordinates after all panels are registered.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    boundary = tight if crop else fig.bbox
    outside: list[str] = []
    for text_obj in fig.findobj(Text):
        if not text_obj.get_visible() or not text_obj.get_text():
            continue
        tick_outside_view = False
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                view_lo, view_hi = sorted(axis.get_view_interval())
                for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                    if text_obj is tick.label1 or text_obj is tick.label2:
                        try:
                            tick_outside_view = float(tick.get_loc()) < view_lo - 1e-10 or float(tick.get_loc()) > view_hi + 1e-10
                        except (TypeError, ValueError):
                            tick_outside_view = False
                        break
                if tick_outside_view:
                    break
            if tick_outside_view:
                break
        # Locators may retain labels for decades just outside a log/linear
        # view.  They are not rendered in the axes and are excluded from the
        # canvas check; every in-view tick label remains subject to bounds QA.
        if tick_outside_view:
            continue
        try:
            box = text_obj.get_window_extent(renderer)
        except Exception:
            continue
        # Matplotlib's tick-label extents can differ from the final tight box
        # by a few device pixels (especially mathtext superscripts).  Reserve
        # a small rasterization tolerance while still catching genuine
        # off-canvas annotations/legends.
        if box.width and box.height and (box.x0 < boundary.x0 - 12 or box.x1 > boundary.x1 + 12 or box.y0 < boundary.y0 - 12 or box.y1 > boundary.y1 + 12):
            outside.append(str(text_obj.get_text()))
    return {"off_canvas_text": outside, "boundary": "tight_export" if crop else "figure_canvas", "tight_bbox_px": [float(tight.x0), float(tight.y0), float(tight.x1), float(tight.y1)], "figure_bbox_px": [float(fig.bbox.x0), float(fig.bbox.y0), float(fig.bbox.x1), float(fig.bbox.y1)]}


def save_figure(fig: Any, base: Path, *, dpi: int = 600, qa_sink: list[dict[str, Any]] | None = None, crop: bool = True) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    qa = _visual_qa(fig, crop=crop)
    qa["file_stem"] = relpath(base)
    if qa_sink is not None:
        qa_sink.append(qa)
    bbox = "tight" if crop else None
    pad_inches = 0.025 if crop else 0.0
    fig.savefig(base.with_suffix(".svg"), format="svg", bbox_inches=bbox, pad_inches=pad_inches, metadata={"Date": None})
    fig.savefig(base.with_suffix(".png"), format="png", dpi=dpi, bbox_inches=bbox, pad_inches=pad_inches)
    plt.close(fig)
    return [relpath(base.with_suffix(".svg")), relpath(base.with_suffix(".png"))]


def interval_record(
    *,
    y: float,
    label: str,
    method: str,
    value: Mapping[str, Any],
    policy: str,
    metric: str,
    target: str,
) -> dict[str, Any]:
    return {
        "y": float(y),
        "label": label,
        "method_key": method,
        "value": float(value["mean"]),
        "ci95_block20": [float(value["low"]), float(value["high"])],
        "policy": policy,
        "metric": metric,
        "target": target,
    }


def draw_si01_fieldwise(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "a", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.7, 2.55), sharey=True)
    records: list[dict[str, Any]] = []
    y_base = np.arange(len(FIELDS))[::-1]
    offsets = np.linspace(-0.23, 0.23, len(STOCHASTIC_ORDER))
    for ax, policy in zip(axes, ("last", "best")):
        for idx, method in enumerate(STOCHASTIC_ORDER):
            vals = [summary_value(data, policy, method, "physical_relative_l2", field_name) for field_name in FIELDS]
            color = style_for(config, method)["color"]
            marker = style_for(config, method)["marker"]
            yy = y_base + offsets[idx]
            ax.errorbar(
                [v["mean"] for v in vals],
                yy,
                xerr=[[v["mean"] - v["low"] for v in vals], [v["high"] - v["mean"] for v in vals]],
                fmt=marker,
                ms=3.6,
                color=color,
                markerfacecolor="white" if method == "A0" else color,
                markeredgewidth=0.7,
                lw=0.65,
                capsize=1.4,
                label=method_label(config, method),
            )
            for y, field_name, value in zip(yy, FIELDS, vals):
                records.append(interval_record(y=float(y), label=field_name, method=method, value=value, policy=policy, metric="physical_relative_l2", target=field_name))
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xlabel("Physical relative $L_2$")
        ax.set_xlim(0.0, 0.86)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.axvline(0, color="#555555", lw=0.45)
        clean_axes(ax)
    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(["T (observed)" if f == "T" else f for f in FIELDS])
    axes[0].set_ylabel("Field")
    handles = [
        Line2D([0], [0], marker=style_for(config, m)["marker"], color=style_for(config, m)["color"], markerfacecolor="white" if m == "A0" else style_for(config, m)["color"], lw=0, ms=4, label=method_label(config, m))
        for m in STOCHASTIC_ORDER
    ]
    if owns_figure:
        axes[1].legend(handles=handles, loc="upper right", fontsize=5.8, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.1)
    else:
        # In the fixed-width entry composite, use the reserved gutter so the
        # key cannot cover a data point in the best-policy facet.
        axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.0), fontsize=5.8, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.0)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.6)
    manifest.add_plot(
        {
            "id": "si01a_fieldwise_physical_l2",
            "role": "fieldwise primary reconstruction error under both checkpoint policies",
            "metric": "physical_relative_l2",
            "normalization": "physical field norm per state; fieldwise summary",
            "statistics": "mean with 95% circular moving-block interval, block length 20, 2,000 replicates, n=1,000 states",
            "source_keys": ["last_summary_metrics", "best_summary_metrics"],
            "coordinates": records,
            "visual_encoding": "method color/marker from shared YAML; horizontal point with block-20 interval; observed T labelled",
            "axes": {"x": "Physical relative L2", "xlim": [0.0, 0.86], "policy_facets": ["last.pt", "best.pt"]},
        }
    )
    return fig


def draw_si01_heatmap(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "b", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.3, 2.42), sharey=True)
    records: list[dict[str, Any]] = []
    vals_all: list[float] = []
    matrices: dict[str, np.ndarray] = {}
    for policy in ("last", "best"):
        mat = np.zeros((len(STOCHASTIC_ORDER), len(FIELDS)), dtype=float)
        full = np.array([summary_value(data, policy, "A0", "physical_relative_l2", f)["mean"] for f in FIELDS])
        for i, method in enumerate(STOCHASTIC_ORDER):
            for j, field_name in enumerate(FIELDS):
                value = summary_value(data, policy, method, "physical_relative_l2", field_name)["mean"]
                mat[i, j] = np.log2(value / full[j])
                records.append({"policy": policy, "method_key": method, "field": field_name, "log2_mean_ratio_to_full": float(mat[i, j]), "mean": float(value), "full_mean": float(full[j])})
        matrices[policy] = mat
        vals_all.extend(mat.ravel().tolist())
    lim = max(abs(min(vals_all)), abs(max(vals_all)))
    lim = max(2.0, math.ceil(lim * 2) / 2)
    for ax, policy in zip(axes, ("last", "best")):
        im = ax.imshow(matrices[policy], aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xticks(range(len(FIELDS)), ["T*" if f == "T" else f for f in FIELDS])
        ax.set_yticks(range(len(STOCHASTIC_ORDER)), [method_label(config, m).replace(" ", "\n", 1) if len(method_label(config, m)) > 16 else method_label(config, m) for m in STOCHASTIC_ORDER])
        ax.tick_params(axis="x", rotation=0)
        for i in range(len(STOCHASTIC_ORDER)):
            for j in range(len(FIELDS)):
                value = matrices[policy][i, j]
                rgba = mpl.colormaps["RdBu_r"](mpl.colors.Normalize(-lim, lim)(value))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=5.2, color="white" if lum < 0.5 else "#202020")
        ax.set_frame_on(False)
        ax.tick_params(length=0, pad=1)
    axes[0].set_ylabel("log$_2$(variant mean / full mean)")
    # Reserve a right-hand gutter for the colorbar explicitly.  Calling
    # tight_layout with a colorbar attached to both axes places it on top of
    # the best-policy panel at the final width.
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.028, pad=0.055, aspect=18)
    cbar.set_label("log$_2$ ratio", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.2, length=2)
    add_panel_tag(axes[0], tag)
    fig.subplots_adjust(left=0.16, right=0.87, bottom=0.20, top=0.90, wspace=0.22)
    # ``Figure.colorbar`` can place the bar between the two facets when the
    # parent is a nested SubFigure.  Pin it after the layout adjustment to a
    # dedicated right gutter so the best-policy heatmap remains fully visible.
    cbar.ax.set_position([0.90, 0.20, 0.022, 0.70])
    manifest.add_plot(
        {
            "id": "si01b_fieldwise_effect_heatmap",
            "role": "fieldwise effect relative to the full ablation reference",
            "metric": "physical_relative_l2",
            "definition": "log2(mean variant / mean A0) computed from fieldwise cohort means; not a mean of statewise ratios",
            "source_keys": ["last_summary_metrics", "best_summary_metrics"],
            "coordinates": records,
            "visual_encoding": "diverging RdBu_r heatmap; zero is the full-model reference; T is identified as observed",
            "axes": {"color_limits": [-lim, lim], "policies": ["last", "best"]},
        }
    )
    return fig


def draw_si01_normalizations(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "c", *, axes: Any = None, figure: Any = None) -> Any:
    # The evaluator stores fieldwise truth-fluctuation-normalized errors but
    # does not define an ``All_fields_mean`` for that normalization.  Plot the
    # recorded field rows directly (four physical fields plus observed T)
    # rather than manufacturing a macro mean from incompatible denominators.
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=2, ncols=2, figsize=(6.6, 4.15), sharey="row")
    records: list[dict[str, Any]] = []
    metrics = [("normalized_relative_l2", "Standardized relative $L_2$"), ("truth_fluctuation_normalized_l2", "Truth-fluctuation normalized $L_2$")]
    y = np.arange(len(FIELDS))[::-1]
    offsets = np.linspace(-0.22, 0.22, len(STOCHASTIC_ORDER))
    for row, (metric, label) in enumerate(metrics):
        for col, policy in enumerate(("last", "best")):
            ax = axes[row, col]
            for idx, method in enumerate(STOCHASTIC_ORDER):
                vals = [summary_value(data, policy, method, metric, field_name) for field_name in FIELDS]
                yy = y + offsets[idx]
                color = style_for(config, method)["color"]
                ax.errorbar(
                    [v["mean"] for v in vals], yy,
                    xerr=[[v["mean"] - v["low"] for v in vals], [v["high"] - v["mean"] for v in vals]],
                    fmt=style_for(config, method)["marker"], ms=2.8, color=color,
                    markerfacecolor="white" if method == "A0" else color,
                    markeredgewidth=0.55, lw=0.5, capsize=1.0,
                    label=method_label(config, method),
                )
                for yv, field_name, value in zip(yy, FIELDS, vals):
                    records.append(interval_record(y=float(yv), label=field_name, method=method, value=value, policy=policy, metric=metric, target=field_name))
            ax.set_title(f"{policy}.pt" if row == 0 else "", loc="left", pad=3)
            ax.set_xlabel(label)
            ax.set_xlim(0, 2.45 if metric == "truth_fluctuation_normalized_l2" else 1.0)
            ax.set_xticks(np.linspace(0, ax.get_xlim()[1], 6))
            clean_axes(ax)
            if col == 0:
                ax.set_yticks(y)
                ax.set_yticklabels(["T (observed)" if f == "T" else f for f in FIELDS])
                ax.set_ylabel("Field")
            else:
                ax.set_yticks(y)
                # Hide only this facet's labels; ``set_yticklabels`` on a
                # shared axis can clear the labels on its left-hand partner.
                ax.tick_params(axis="y", labelleft=False)
    handles = [Line2D([0], [0], marker=style_for(config, m)["marker"], color=style_for(config, m)["color"], markerfacecolor="white" if m == "A0" else style_for(config, m)["color"], lw=0, ms=3.5, label=method_label(config, m)) for m in STOCHASTIC_ORDER]
    axes[0, 1].legend(handles=handles, loc="upper right", fontsize=5.8, handletextpad=0.25, labelspacing=0.15, borderaxespad=0.1)
    add_panel_tag(axes[0, 0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.1, h_pad=1.0)
    manifest.add_plot({"id": "si01c_normalization_checks", "role": "fieldwise standardized and truth-fluctuation normalizations retained alongside physical L2", "metrics": [m[0] for m in metrics], "source_keys": ["last_summary_metrics", "best_summary_metrics"], "coordinates": records, "visual_encoding": "method color/marker; fieldwise horizontal point and block-20 interval; observed T labelled", "axes": {"x": "fieldwise relative L2", "xlim_standardized": [0, 1.0], "xlim_truth_fluctuation": [0, 2.45], "policy_facets": ["last.pt", "best.pt"]}})
    return fig


def draw_si01_local_detail(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "d", *, axes: Any = None, figure: Any = None) -> Any:
    if axes is None:
        fig, ax = plt.subplots(figsize=(3.35, 2.55))
        owns_figure = True
    else:
        fig = figure if figure is not None else np.asarray(axes, dtype=object).flat[0].figure
        ax = np.asarray(axes, dtype=object).flat[0]
        owns_figure = False
    methods = ["A0", "A3"]
    vals = [summary_value(data, "last", m, "physical_relative_l2", "CH4") for m in methods]
    y = np.arange(len(methods))[::-1]
    records = []
    for yy, method, value in zip(y, methods, vals):
        color = style_for(config, method)["color"]
        ax.errorbar(value["mean"], yy, xerr=[[value["mean"] - value["low"]], [value["high"] - value["mean"]]], fmt=style_for(config, method)["marker"], ms=5, color=color, markerfacecolor="white" if method == "A0" else color, markeredgewidth=0.8, lw=0.7, capsize=2)
        records.append(interval_record(y=float(yy), label=method_label(config, method), method=method, value=value, policy="last", metric="physical_relative_l2", target="CH4"))
    delta = 100 * (vals[1]["mean"] / vals[0]["mean"] - 1)
    # The effect belongs to the no-local row (A3, y=0), and the text is kept
    # inside the declared x/y limits so it cannot be clipped in the SVG.
    ax.annotate(f"+{delta:.1f}%", xy=(vals[1]["mean"], y[1]), xytext=(0.157, 0.35), ha="right", fontsize=6.0, color="#404040", arrowprops={"arrowstyle": "-", "lw": 0.45, "color": "#777777"})
    ax.set_yticks(y)
    ax.set_yticklabels([method_label(config, m) for m in methods])
    ax.set_xlabel("CH$_4$ physical relative $L_2$")
    ax.set_title("Local-conditioning removal", loc="left", pad=3)
    ax.set_xlim(0.035, 0.16)
    ax.text(0.04, -0.24, "last.pt; scalar error does not imply a physical length scale", transform=ax.transAxes, fontsize=5.8, color="#555555")
    clean_axes(ax)
    add_panel_tag(ax, tag)
    if owns_figure:
        fig.tight_layout()
    manifest.add_plot({"id": "si01d_local_conditioning_ch4_detail", "role": "reported local-conditioning sensitivity detail", "metric": "physical_relative_l2", "source_keys": ["last_summary_metrics"], "coordinates": records, "derived_comparison": {"definition": "100*(A3/A0-1) from cohort means", "percent": float(delta), "formatted": f"+{delta:.1f}%"}, "visual_encoding": "horizontal point and block-20 interval", "axes": {"x": "CH4 physical relative L2", "xlim": [0.035, 0.16]}})
    return fig


def draw_si02_highband_l2(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "a", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.6, 2.55), sharey=True)
    records = []
    y_base = np.arange(len(FIELDS))[::-1]
    for ax, policy in zip(axes, ("last", "best")):
        for method, offset in (("A0", -0.09), ("A4", 0.09)):
            values = [hf_value(data, policy, method, field_name, "highband_error_relative_l2") for field_name in FIELDS]
            yy = y_base + offset
            color = style_for(config, method)["color"]
            ax.errorbar([v["mean"] for v in values], yy, xerr=[[v["mean"] - v["low"] for v in values], [v["high"] - v["mean"] for v in values]], fmt=style_for(config, method)["marker"], ms=4, color=color, markerfacecolor="white" if method == "A0" else color, markeredgewidth=0.75, lw=0.7, capsize=1.6, label="RFF (full)" if method == "A0" else "IID")
            for y, field_name, value in zip(yy, FIELDS, values):
                records.append(interval_record(y=float(y), label=field_name, method=method, value=value, policy=policy, metric="highband_error_relative_l2", target=field_name))
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xlabel("High-band relative $L_2$")
        ax.set_xlim(0, 2.25)
        clean_axes(ax)
    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(["T (observed)" if f == "T" else f for f in FIELDS])
    axes[0].set_ylabel("Field")
    axes[1].legend(loc="lower right", fontsize=5.8, handletextpad=0.35, labelspacing=0.3)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.4)
    manifest.add_plot({"id": "si02a_all_field_highband_l2", "role": "all-field phase-sensitive high-band residual", "metric": "highband_error_relative_l2", "source_keys": ["high_frequency_summary"], "coordinates": records, "statistics": "mean with 95% block-20 interval; n=1,000 states per method/policy", "visual_encoding": "RFF hollow red circles; IID filled amber diamonds; policy facets", "axes": {"x": "High-band relative L2", "ideal": 0, "xlim": [0, 2.25]}})
    return fig


def draw_si02_power(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "b", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.6, 2.55), sharey=True)
    records = []
    y_base = np.arange(len(FIELDS))[::-1]
    metric = "canonical_shellmean_high_energy_ratio"
    for ax, policy in zip(axes, ("last", "best")):
        for method, offset in (("A0", -0.09), ("A4", 0.09)):
            values = [hf_value(data, policy, method, field_name, metric) for field_name in FIELDS]
            yy = y_base + offset
            color = style_for(config, method)["color"]
            ax.errorbar([v["mean"] for v in values], yy, xerr=[[v["mean"] - v["low"] for v in values], [v["high"] - v["mean"] for v in values]], fmt=style_for(config, method)["marker"], ms=4, color=color, markerfacecolor="white" if method == "A0" else color, markeredgewidth=0.75, lw=0.7, capsize=1.6, label="RFF (full)" if method == "A0" else "IID")
            for y, field_name, value in zip(yy, FIELDS, values):
                records.append(interval_record(y=float(y), label=field_name, method=method, value=value, policy=policy, metric=metric, target=field_name))
        ax.axvline(1, ls="--", color="#555555", lw=0.65)
        ax.text(1.01, 1.02, "truth", transform=ax.get_xaxis_transform(), fontsize=5.3, color="#555555", va="bottom")
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xlabel("High-band power / truth")
        ax.set_xlim(0, 3.8)
        clean_axes(ax)
    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(["T (observed)" if f == "T" else f for f in FIELDS])
    axes[0].set_ylabel("Field")
    axes[1].legend(loc="upper right", fontsize=5.8, handletextpad=0.35, labelspacing=0.3)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.4)
    manifest.add_plot({"id": "si02b_all_field_canonical_power", "role": "canonical shell-mean high-band power ratio", "metric": metric, "definition": "per-state shell-mean spectrum integrated with existing trapezoidal weighting over strict retained high band; not mode-sum", "source_keys": ["high_frequency_summary"], "coordinates": records, "statistics": "mean with 95% block-20 interval; n=1,000 states per method/policy", "visual_encoding": "RFF hollow red circles; IID filled amber diamonds; truth=1 dashed guide", "axes": {"x": "High-band power / truth", "ideal": 1, "xlim": [0, 3.8]}})
    return fig


def draw_si03_spectra(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "a", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.6, 2.55), sharey=True)
    records = []
    for ax, policy in zip(axes, ("last", "best")):
        for method, ls in (("A0", "-"), ("A4", "--")):
            frame = data.population_spectra[(data.population_spectra["policy"] == policy) & (data.population_spectra["method"] == method) & (data.population_spectra["field"] == "U1")].sort_values("shell_index")
            if frame.empty:
                raise KeyError(f"no population spectrum rows for {policy}/{method}/U1")
            x = pd.to_numeric(frame["wavenumber"], errors="coerce").to_numpy(float) / pd.to_numeric(frame["kmax"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(frame["median"], errors="coerce").to_numpy(float)
            low = pd.to_numeric(frame["q25"], errors="coerce").to_numpy(float)
            high = pd.to_numeric(frame["q75"], errors="coerce").to_numpy(float)
            color = style_for(config, method)["color"]
            ax.plot(x, y, color=color, ls=ls, lw=1.0, label="RFF (full)" if method == "A0" else "IID")
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
            records.append({"policy": policy, "method_key": method, "field": "U1", "x_wavenumber_over_kmax": x.tolist(), "median": y.tolist(), "q25": low.tolist(), "q75": high.tolist(), "high_band_mask": frame["high_band"].astype(bool).tolist()})
        ax.axhline(1, ls="--", color="#555555", lw=0.65)
        ax.axvspan(2 / 3, 1, color="#E9EEF3", alpha=0.7, zorder=0)
        ax.text(0.675, 0.94, "strict high band", transform=ax.get_xaxis_transform(), fontsize=5.1, color="#52606D")
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xlabel("Retained index-space wavenumber / $k_{max}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(3.8, float(np.nanmax(data.population_spectra["q75"]))*1.04))
        clean_axes(ax)
    axes[0].set_ylabel("Shell-power ratio")
    axes[1].legend(loc="upper left", fontsize=5.8, handletextpad=0.35, labelspacing=0.3)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.25)
    manifest.add_plot({"id": "si03a_u1_population_spectra", "role": "population shell-power ratio distribution across retained index-space", "metric": "population_U1_spectra median/IQR", "source_keys": ["population_U1_spectra"], "statistics": "per-state median and IQR; shading is state dispersion, not a confidence band; n=1,000 per shell/method/policy", "coordinates": records, "visual_encoding": "RFF solid red; IID dashed amber; IQR alpha=0.15; truth=1; strict high band shaded", "axes": {"x": "wavenumber/kmax", "xlim": [0, 1], "high_band": ">2/3", "y_ideal": 1}})
    return fig


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def draw_si03_ecdf(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "b", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=2, ncols=2, figsize=(6.45, 3.9), sharex="col", sharey="row")
    records = []
    metrics = [("reconstruction_to_truth_high_energy_ratio", "Mode-sum high-band power / truth", "power"), ("highband_error_relative_l2", "High-band relative $L_2$", "residual")]
    for col, (metric, xlabel, role) in enumerate(metrics):
        for row, policy in enumerate(("last", "best")):
            ax = axes[row, col]
            for method, ls in (("A0", "-"), ("A4", "--")):
                frame = data.hf_per_state[(data.hf_per_state["policy"] == policy) & (data.hf_per_state["method"] == method) & (data.hf_per_state["field"] == "U1") & (data.hf_per_state["metric"] == metric)].sort_values(["snapshot", "time_index"])
                x, y = _ecdf(pd.to_numeric(frame["value"], errors="coerce").to_numpy(float))
                color = style_for(config, method)["color"]
                ax.step(x, y, where="post", color=color, ls=ls, lw=0.9, label="RFF (full)" if method == "A0" else "IID")
                records.append({"policy": policy, "method_key": method, "field": "U1", "metric": metric, "x": x.tolist(), "ecdf": y.tolist(), "n": int(len(x))})
            if role == "power":
                ax.axvline(1, ls=":", color="#555555", lw=0.55)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("ECDF" if col == 0 else "")
            ax.set_xlabel(xlabel if row == 1 else "")
            ax.set_title(f"{policy}.pt" if col == 0 else "", loc="left", pad=2.5)
            clean_axes(ax)
        axes[0, col].legend(loc="lower right", fontsize=5.8, handletextpad=0.3, labelspacing=0.2)
    add_panel_tag(axes[0, 0], tag)
    if owns_figure:
        fig.tight_layout(h_pad=1.0, w_pad=1.2)
    manifest.add_plot({"id": "si03b_u1_mode_sum_ecdf", "role": "empirical distributions of mode-sum power and high-band residual", "metrics": [m[0] for m in metrics], "source_keys": ["high_frequency_per_state"], "statistics": "ECDF over 1,000 states; this mode-sum power is separate from canonical trapezoidal ratio", "coordinates": records, "visual_encoding": "RFF solid red; IID dashed amber; dotted power guide at truth=1", "axes": {"power_ideal": 1, "residual_ideal": 0, "y": "ECDF"}})
    return fig


def draw_si03_hann(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "c", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.2, 2.6), sharey=True)
    records = []
    metrics = [("hann_reconstruction_to_truth_high_energy_ratio", "Hann power / truth"), ("hann_highband_error_relative_l2", "Hann high-band relative $L_2$")]
    y = np.arange(2)[::-1]
    for ax, policy in zip(axes, ("last", "best")):
        for idx, (metric, label) in enumerate(metrics):
            for method, offset in (("A0", -0.11), ("A4", 0.11)):
                # Hann robustness uses the same diagnostic IDs as the
                # untapered high-frequency source (without the ``hann_``
                # prefix); the window column identifies the estimator.
                diagnostic = metric.removeprefix("hann_")
                rows = data.hann_summary[(data.hann_summary["policy"] == policy) & (data.hann_summary["method"] == method) & (data.hann_summary["field"] == "U1") & (data.hann_summary["window"] == "hann") & (data.hann_summary["diagnostic"] == diagnostic)]
                if len(rows) != 1:
                    raise KeyError(f"missing Hann row {policy}/{method}/{metric}")
                row = rows.iloc[0]
                value = {"mean": safe_float(row["mean"]), "low": safe_float(row["block20_ci95_low"]), "high": safe_float(row["block20_ci95_high"])}
                yy = y[idx] + offset
                color = style_for(config, method)["color"]
                ax.errorbar(value["mean"], yy, xerr=[[value["mean"] - value["low"]], [value["high"] - value["mean"]]], fmt=style_for(config, method)["marker"], ms=3.8, color=color, markerfacecolor="white" if method == "A0" else color, markeredgewidth=0.7, lw=0.65, capsize=1.5)
                records.append(interval_record(y=float(yy), label=label, method=method, value=value, policy=policy, metric=metric, target="U1"))
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xlabel("Hann diagnostic")
        ax.set_xlim(0, 12.8 if policy == "best" else 12.3)
        clean_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(["Power ratio", "High-band relative $L_2$"])
    axes[0].set_ylabel("Hann metric")
    axes[1].legend(handles=[Line2D([0], [0], marker=style_for(config, m)["marker"], color=style_for(config, m)["color"], markerfacecolor="white" if m == "A0" else style_for(config, m)["color"], lw=0, ms=4, label="RFF (full)" if m == "A0" else "IID") for m in ("A0", "A4")], loc="lower right", fontsize=5.8, handletextpad=0.35, labelspacing=0.25)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.4)
    prevalence = {}
    for policy in ("last", "best"):
        for method in ("A0", "A4"):
            rows = data.hf_summary[(data.hf_summary["policy"] == policy) & (data.hf_summary["method"] == method) & (data.hf_summary["field"] == "U1") & (data.hf_summary["metric"] == "reconstruction_to_truth_high_energy_ratio")]
            if len(rows) == 1:
                prevalence[f"{policy}_{method}"] = {"fraction_gt_1": safe_float(rows.iloc[0]["fraction_gt_1"]), "fraction_gt_2": safe_float(rows.iloc[0]["fraction_gt_2"])}
    manifest.add_plot({"id": "si03c_u1_hann_sensitivity", "role": "Hann-window sensitivity kept separate from primary untapered estimators", "metrics": [m[0] for m in metrics], "source_keys": ["U1_hann_robustness_summary", "high_frequency_summary"], "statistics": "mean with 95% block-20 interval; n=1,000", "coordinates": records, "prevalence_mode_sum_primary": prevalence, "visual_encoding": "RFF hollow red; IID filled amber; policy facets", "axes": {"x": "Hann diagnostic", "primary_untapered": False}})
    return fig


def draw_si04_metric(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, metric: str, title: str, tag: str, *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.55, 2.75), sharey=True)
    records = []
    y = np.arange(len(PAIR_TARGETS))[::-1]
    for ax, policy in zip(axes, ("last", "best")):
        for method in STOCHASTIC_ORDER:
            values = [summary_value(data, policy, method, metric, target) for target in PAIR_TARGETS]
            color = style_for(config, method)["color"]
            ax.plot([v["mean"] for v in values], y, linestyle="none", marker=style_for(config, method)["marker"], ms=3.8, color=color, markerfacecolor="white" if method == "A0" else color, markeredgewidth=0.7, label=method_label(config, method))
            for yy, target, value in zip(y, PAIR_TARGETS, values):
                ax.errorbar(value["mean"], yy, xerr=[[value["mean"] - value["low"]], [value["high"] - value["mean"]]], fmt="none", ecolor=color, lw=0.55, capsize=1.3)
                records.append(interval_record(y=float(yy), label=target, method=method, value=value, policy=policy, metric=metric, target=target))
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xlabel(title)
        ax.set_xlim(0, 0.62 if "overflow" in metric else 0.44)
        clean_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(["T–U1", "CH$_4$–U1", "p–U1"])
    axes[0].set_ylabel("Coupled pair")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.8, handletextpad=0.28, labelspacing=0.2, ncol=1, borderaxespad=0.0)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.35)
    manifest.add_plot({"id": f"si04_{metric}", "role": title, "metric": metric, "source_keys": ["last_summary_metrics", "best_summary_metrics"], "statistics": "mean with 95% block-20 interval; n=1,000; common paper edges", "coordinates": records, "visual_encoding": "method color/marker; policy facets; JSD base 2", "axes": {"x": title, "pair_targets": PAIR_TARGETS, "histogram_definition": "64x64 original edges" if "overflow" not in metric else "66x66 overflow-retaining extension"}})
    return fig


def draw_si04_retention(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "c", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.45, 2.75), sharey=True)
    records = []
    y = np.arange(len(PAIR_TARGETS))[::-1]
    for ax, policy in zip(axes, ("last", "best")):
        for method in STOCHASTIC_ORDER:
            pred = [summary_value(data, policy, method, "joint_pdf_reconstruction_retained_fraction", target) for target in PAIR_TARGETS]
            truth = [summary_value(data, policy, method, "joint_pdf_truth_retained_fraction", target) for target in PAIR_TARGETS]
            for values, marker, label, alpha in ((pred, style_for(config, method)["marker"], method_label(config, method), 1.0), (truth, "_", "truth", 0.55)):
                x = [v["mean"] for v in values]
                color = style_for(config, method)["color"] if label != "truth" else "#555555"
                if label == "truth":
                    ax.plot(x, y + 0.12, marker=marker, linestyle="none", ms=4, color=color, alpha=alpha, label=label if method == STOCHASTIC_ORDER[0] else None)
                else:
                    ax.plot(x, y - 0.12, marker=marker, linestyle="none", ms=3.8, color=color, markerfacecolor="white" if method == "A0" else color, markeredgewidth=0.7, label=label)
                for yy, target, value in zip(y, PAIR_TARGETS, values):
                    records.append(interval_record(y=float(yy + (0.12 if label == "truth" else -0.12)), label=target, method=method if label != "truth" else "truth", value=value, policy=policy, metric="joint_pdf_truth_retained_fraction" if label == "truth" else "joint_pdf_reconstruction_retained_fraction", target=target))
        ax.axvline(0.99, ls=":", color="#555555", lw=0.55)
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xlabel("Retained predicted/truth pair fraction")
        # Include the complete interval for the low-retention A3 CH4-U1 row
        # (its lower block-20 endpoint is about 0.441).
        ax.set_xlim(0.42, 1.01)
        clean_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(["T–U1", "CH$_4$–U1", "p–U1"])
    axes[0].set_ylabel("Coupled pair")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.8, handletextpad=0.25, labelspacing=0.18, borderaxespad=0.0)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.35)
    manifest.add_plot({"id": "si04c_coupling_retention", "role": "retained-pair context for truncated and overflow JSD", "metrics": ["joint_pdf_reconstruction_retained_fraction", "joint_pdf_truth_retained_fraction"], "source_keys": ["last_summary_metrics", "best_summary_metrics"], "statistics": "mean with 95% block-20 interval; n=1,000; retention is not calibration", "coordinates": records, "visual_encoding": "predicted points offset below pair row; truth ticks above; method color; 0.99 guide", "axes": {"x": "retention fraction", "xlim": [0.42, 1.01]}})
    return fig


def draw_si05_checkpoint(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "a", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.5, 2.85))
    records = []
    sensitivity = data.sensitivity
    entries = [("physical_relative_l2", "Unobserved_mean", "Macro physical relative $L_2$"), ("joint_pdf_jsd_with_overflow_base2", "CH4-U1", "CH$_4$–U1 overflow JSD")]
    y = np.arange(len(STOCHASTIC_ORDER))[::-1]
    for ax, (metric, target, xlabel) in zip(axes, entries):
        for method in STOCHASTIC_ORDER:
            rows = sensitivity[(sensitivity["method"] == method) & (sensitivity["metric"] == metric) & (sensitivity["target"] == target)]
            if len(rows) != 1:
                # The source uses coupling target strings for the second audit;
                # leave a visible gap when a metric is not recorded rather than
                # deriving an endpoint difference from unrelated summaries.
                continue
            row = rows.iloc[0]
            value = {"mean": safe_float(row["best_minus_last"]), "low": safe_float(row["block20_ci95_low"]), "high": safe_float(row["block20_ci95_high"])}
            color = style_for(config, method)["color"]
            ax.errorbar(value["mean"], y[list(STOCHASTIC_ORDER).index(method)], xerr=[[value["mean"] - value["low"]], [value["high"] - value["mean"]]], fmt=style_for(config, method)["marker"], ms=3.8, color=color, markerfacecolor="white" if method == "A0" else color, markeredgewidth=0.7, lw=0.65, capsize=1.5)
            records.append({"policy_comparison": "best-minus-last", "method_key": method, "metric": metric, "target": target, "value": value["mean"], "ci95_block20": [value["low"], value["high"]], "n": int(row["n_paired"])})
        ax.axvline(0, color="#555555", lw=0.55)
        ax.set_xlabel(xlabel)
        ax.set_title("best − last", loc="left", pad=3)
        clean_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([method_label(config, m) for m in STOCHASTIC_ORDER])
    axes[0].set_ylabel("Configuration")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([method_label(config, m) for m in STOCHASTIC_ORDER])
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.5)
    manifest.add_plot({"id": "si05a_checkpoint_sensitivity", "role": "best-minus-last checkpoint-selection sensitivity", "source_keys": ["checkpoint_sensitivity"], "coordinates": records, "statistics": "paired block-20 intervals from saved policy comparison; policy values are not pooled", "visual_encoding": "horizontal signed point/interval; zero guide; fixed stochastic order", "axes": {"x": "best-minus-last", "metrics": [e[0] for e in entries]}})
    return fig


def draw_si05_blocks(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "b", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.45, 2.75))
    records = []
    # These intervals are the saved paired best-minus-last comparisons.  Keep
    # the same two sensitive quantities as SI05a while exposing every stored
    # circular-block interval (5, 20 and 50 states).
    comparisons = [
        ("physical_relative_l2", "Unobserved_mean", "Macro physical relative $L_2$ (best − last)"),
        ("joint_pdf_jsd_with_overflow_base2", "CH4-U1", "CH$_4$–U1 overflow JSD (best − last)"),
    ]
    blocks = (5, 20, 50)
    for ax, (metric, target, xlabel) in zip(axes, comparisons):
        interval_lows: list[float] = []
        interval_highs: list[float] = []
        for method_index, method in enumerate(STOCHASTIC_ORDER):
            rows = data.sensitivity[(data.sensitivity["method"] == method) & (data.sensitivity["metric"] == metric) & (data.sensitivity["target"] == target)]
            if len(rows) != 1:
                raise KeyError(f"missing checkpoint sensitivity row {method}/{metric}/{target}")
            row = rows.iloc[0]
            value = safe_float(row["best_minus_last"])
            color = style_for(config, method)["color"]
            # Separate nearly coincident method intervals by a small
            # horizontal offset around each block-length tick.
            xs = np.arange(len(blocks), dtype=float) + (method_index - 2) * 0.035
            lows = [safe_float(row[f"block{block}_ci95_low"]) for block in blocks]
            highs = [safe_float(row[f"block{block}_ci95_high"]) for block in blocks]
            ax.plot(xs, [value] * len(blocks), color=color, marker=style_for(config, method)["marker"], ms=3.4, lw=0.75, label=method_label(config, method))
            for xx, low, high, block in zip(xs, lows, highs, blocks):
                interval_lows.append(low)
                interval_highs.append(high)
                ax.vlines(xx, low, high, color=color, lw=0.55)
                records.append({
                    "policy_comparison": "best-minus-last",
                    "method_key": method,
                    "metric": metric,
                    "target": target,
                    "block_length": block,
                    "x_position": float(xx),
                    "value": value,
                    "ci95_block": [low, high],
                    "last_mean": safe_float(row["last_mean"]),
                    "best_mean": safe_float(row["best_mean"]),
                    "n": int(row["n_paired"]),
                    "source_key": "checkpoint_sensitivity",
                })
        ax.axhline(0, color="#555555", lw=0.55)
        ax.set_xticks(range(len(blocks)), [str(block) for block in blocks])
        ax.set_xlabel("Circular block length (states)")
        ax.set_ylabel(xlabel)
        ax.set_xlim(-0.15, len(blocks) - 0.85)
        lo = min(interval_lows)
        hi = max(interval_highs)
        pad = max((hi - lo) * 0.12, 1e-4 if metric == "physical_relative_l2" else 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
        clean_axes(ax)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.8, handletextpad=0.25, labelspacing=0.17, borderaxespad=0.0)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.3)
    manifest.add_plot({"id": "si05b_block_length_sensitivity", "role": "paired best-minus-last sensitivity across circular block lengths", "source_keys": ["checkpoint_sensitivity"], "coordinates": records, "statistics": "saved paired best-minus-last mean with source block-5/20/50 intervals; n=1,000", "visual_encoding": "method-colored horizontal line across block lengths with block-specific vertical intervals and zero guide", "axes": {"x": "block length in sorted held-out states", "blocks": list(blocks), "metrics": [metric for metric, _, _ in comparisons], "policy_comparison": "best-minus-last"}})
    return fig


def _selected_epoch(provenance: Mapping[str, Any], policy: str) -> int | None:
    if not provenance:
        return None
    checkpoints = provenance.get("checkpoints", {})
    value = checkpoints.get(policy, {}) if isinstance(checkpoints, Mapping) else {}
    if isinstance(value, Mapping) and value.get("epoch") is not None:
        return int(value["epoch"])
    if provenance.get("checkpoint_epoch") is not None and policy == "last":
        return int(provenance["checkpoint_epoch"])
    return None


def draw_si05_histories(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "c", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.7, 2.85), sharex=True)
    records = []
    for method in STOCHASTIC_ORDER:
        history = data.histories.get(method)
        if history is None:
            continue
        epoch = pd.to_numeric(history["epoch"], errors="coerce").to_numpy(float)
        train = pd.to_numeric(history["train_loss"], errors="coerce").to_numpy(float)
        val = pd.to_numeric(history["val_loss"], errors="coerce").to_numpy(float)
        color = style_for(config, method)["color"]
        label = method_label(config, method)
        last_mask = np.isfinite(train)
        val_mask = np.isfinite(val)
        # Validation is recorded periodically, with NaNs between evaluations.
        # Mask first so Matplotlib connects only adjacent recorded validation
        # samples; this does not interpolate or manufacture missing epochs.
        axes[0].plot(epoch[last_mask], train[last_mask], color=color, lw=0.55, alpha=0.86, label=label)
        axes[1].plot(epoch[val_mask], val[val_mask], color=color, lw=0.75, alpha=0.9, label=label)
        for kind, xs, ys in (("train", epoch[last_mask], train[last_mask]), ("validation", epoch[val_mask], val[val_mask])):
            records.append({"method_key": method, "series": kind, "x_epoch": xs.tolist(), "y_loss": ys.tolist(), "n": int(len(xs)), "source_key": f"history_{method}"})
        for ax, policy, marker in ((axes[0], "last", "o"), (axes[1], "best", "D")):
            selected = _selected_epoch(data.provenance.get(f"last_{method}", {}), policy)
            if selected is not None:
                row = history.loc[pd.to_numeric(history["epoch"], errors="coerce") == selected]
                if not row.empty:
                    y_value = float(row["train_loss"].iloc[0] if ax is axes[0] else row["val_loss"].iloc[0])
                    if math.isfinite(y_value):
                        ax.scatter([selected], [y_value], s=17, marker=marker, color=color, edgecolor="white", linewidth=0.35, zorder=5)
                        records.append({"method_key": method, "series": "selected_checkpoint_marker", "policy": policy, "x_epoch": int(selected), "y_loss": y_value, "source_key": f"history_{method}"})
    axes[0].set_title("Recorded training objective", loc="left", pad=3)
    axes[1].set_title("Recorded validation objective", loc="left", pad=3)
    axes[0].set_ylabel("Velocity loss")
    axes[1].set_ylabel("Velocity loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        # Keep the recorded range readable and suppress unused logarithmic
        # decade labels below the observed loss values.
        ax.set_ylim(0.08, 2.0)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=4))
        ax.yaxis.set_minor_formatter(NullFormatter())
        clean_axes(ax)
    axes[0].legend(loc="upper right", fontsize=5.8, handletextpad=0.25, labelspacing=0.14)
    if not data.histories:
        axes[0].text(0.5, 0.5, "No recorded histories found", transform=axes[0].transAxes, ha="center", va="center", fontsize=6)
        axes[1].text(0.5, 0.5, "History panel unavailable", transform=axes[1].transAxes, ha="center", va="center", fontsize=6)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.2)
    manifest.add_plot({"id": "si05c_recorded_training_histories", "role": "actual recorded train/validation loss histories", "source_keys": [f"history_{m}" for m in STOCHASTIC_ORDER if m in data.histories], "coordinates": records, "statistics": "raw recorded history points; no smoothing, normalization, truncation or endpoint reconstruction", "visual_encoding": "method color; train/validation separate axes; selected checkpoint markers: circle=last, diamond=best", "axes": {"x": "epoch", "y": "loss (log scale)", "missing_methods": [m for m in STOCHASTIC_ORDER if m not in data.histories]}})
    return fig


def draw_si06_macro(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "a", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.45, 2.75), sharey=True)
    records = []
    methods = ["A0", "A1"]
    y = np.arange(len(methods))[::-1]
    for ax, policy in zip(axes, ("last", "best")):
        for method in ("A0", "A1"):
            value = summary_value(data, policy, method, "physical_relative_l2", "Unobserved_mean")
            # One method occupies each row, so align the points to the
            # labelled objective rows rather than offsetting them.
            yy = y[methods.index(method)]
            color = style_for(config, method)["color"]
            ax.errorbar(value["mean"], yy, xerr=[[value["mean"] - value["low"]], [value["high"] - value["mean"]]], fmt=style_for(config, method)["marker"], ms=4, color=color, markerfacecolor="white" if method == "A0" else color, markeredgewidth=0.75, lw=0.7, capsize=1.6)
            records.append(interval_record(y=float(yy), label=method_label(config, method), method=method, value=value, policy=policy, metric="physical_relative_l2", target="Unobserved_mean"))
        advantage = 100 * (summary_value(data, policy, "A1", "physical_relative_l2", "Unobserved_mean")["mean"] / summary_value(data, policy, "A0", "physical_relative_l2", "Unobserved_mean")["mean"] - 1)
        ax.text(0.98, 0.06, f"deterministic {abs(advantage):.1f}% lower", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.3, color="#555555")
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xlabel("Unobserved-field physical relative $L_2$")
        ax.set_xlim(0.04, 0.14)
        clean_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(["Full model", "Deterministic regression"])
    axes[0].set_ylabel("Objective")
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.3)
    manifest.add_plot({"id": "si06a_deterministic_macro_control", "role": "separate deterministic objective control", "metric": "physical_relative_l2", "source_keys": ["last_summary_metrics", "best_summary_metrics"], "coordinates": records, "derived_comparison": {"last_deterministic_minus_full_percent": float(100 * (summary_value(data, "last", "A1", "physical_relative_l2", "Unobserved_mean")["mean"] / summary_value(data, "last", "A0", "physical_relative_l2", "Unobserved_mean")["mean"] - 1)), "best_deterministic_minus_full_percent": float(100 * (summary_value(data, "best", "A1", "physical_relative_l2", "Unobserved_mean")["mean"] / summary_value(data, "best", "A0", "physical_relative_l2", "Unobserved_mean")["mean"] - 1))}, "visual_encoding": "horizontal point/interval; A0 hollow red, A1 filled slate", "axes": {"x": "unobserved physical relative L2"}})
    return fig


def draw_si06_fields(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "b", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.45, 2.75), sharey=True)
    records = []
    y = np.arange(len(FIELDS))[::-1]
    for ax, policy in zip(axes, ("last", "best")):
        for method, offset in (("A0", -0.1), ("A1", 0.1)):
            vals = [summary_value(data, policy, method, "physical_relative_l2", field_name) for field_name in FIELDS]
            yy = y + offset
            color = style_for(config, method)["color"]
            ax.errorbar([v["mean"] for v in vals], yy, xerr=[[v["mean"] - v["low"] for v in vals], [v["high"] - v["mean"] for v in vals]], fmt=style_for(config, method)["marker"], ms=3.6, color=color, markerfacecolor="white" if method == "A0" else color, markeredgewidth=0.7, lw=0.65, capsize=1.3)
            for yv, field_name, value in zip(yy, FIELDS, vals):
                records.append(interval_record(y=float(yv), label=field_name, method=method, value=value, policy=policy, metric="physical_relative_l2", target=field_name))
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xlabel("Physical relative $L_2$")
        ax.set_xlim(0, 0.85)
        clean_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(["T (observed)" if f == "T" else f for f in FIELDS])
    axes[0].set_ylabel("Field")
    axes[1].legend(handles=[Line2D([0], [0], marker=style_for(config, m)["marker"], color=style_for(config, m)["color"], markerfacecolor="white" if m == "A0" else style_for(config, m)["color"], lw=0, ms=4, label=method_label(config, m)) for m in ("A0", "A1")], loc="lower right", fontsize=5.8, handletextpad=0.3, labelspacing=0.2)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.35)
    manifest.add_plot({"id": "si06b_deterministic_fieldwise_control", "role": "fieldwise deterministic-versus-full control", "metric": "physical_relative_l2", "source_keys": ["last_summary_metrics", "best_summary_metrics"], "coordinates": records, "visual_encoding": "A0 hollow red; deterministic A1 filled slate; policy facets", "axes": {"x": "physical relative L2"}})
    return fig


def draw_si06_coupling(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, tag: str = "c", *, axes: Any = None, figure: Any = None) -> Any:
    fig, axes, owns_figure = _adopt_axes(axes, figure, nrows=1, ncols=2, figsize=(6.55, 2.85), sharey=True)
    records = []
    metrics = [("joint_pdf_jsd_base2", "Paper JSD"), ("joint_pdf_jsd_with_overflow_base2", "Overflow JSD")]
    y = np.arange(len(PAIR_TARGETS))[::-1]
    for ax, policy in zip(axes, ("last", "best")):
        for idx, (metric, label) in enumerate(metrics):
            for method, offset in (("A0", -0.12 if idx == 0 else 0.12), ("A1", 0.12 if idx == 0 else -0.12)):
                vals = [summary_value(data, policy, method, metric, target) for target in PAIR_TARGETS]
                yy = y + offset
                color = style_for(config, method)["color"]
                marker = style_for(config, method)["marker"] if idx == 0 else "x"
                ax.errorbar([v["mean"] for v in vals], yy, xerr=[[v["mean"] - v["low"] for v in vals], [v["high"] - v["mean"] for v in vals]], fmt=marker, ms=3.7, color=color, markerfacecolor="white" if method == "A0" and idx == 0 else color, markeredgewidth=0.7, lw=0.55, capsize=1.3)
                for yv, target, value in zip(yy, PAIR_TARGETS, vals):
                    records.append(interval_record(y=float(yv), label=target, method=method, value=value, policy=policy, metric=metric, target=target))
        ax.set_title(f"{policy}.pt", loc="left", pad=3)
        ax.set_xlabel("JSD (base 2)")
        ax.set_xlim(0, 0.62)
        clean_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(["T–U1", "CH$_4$–U1", "p–U1"])
    axes[0].set_ylabel("Coupled pair")
    axes[1].legend(handles=[Line2D([0], [0], marker=style_for(config, "A0")["marker"], color=style_for(config, "A0")["color"], lw=0, ms=4, label="Full / paper JSD"), Line2D([0], [0], marker="x", color=style_for(config, "A0")["color"], lw=0, ms=4, label="Full / overflow JSD"), Line2D([0], [0], marker=style_for(config, "A1")["marker"], color=style_for(config, "A1")["color"], lw=0, ms=4, label="Deterministic / paper JSD"), Line2D([0], [0], marker="x", color=style_for(config, "A1")["color"], lw=0, ms=4, label="Deterministic / overflow JSD")], loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.8, handletextpad=0.28, labelspacing=0.16, borderaxespad=0.0)
    add_panel_tag(axes[0], tag)
    if owns_figure:
        fig.tight_layout(w_pad=1.3)
    manifest.add_plot({"id": "si06c_deterministic_coupling_control", "role": "deterministic coupling diagnostics with range companions", "metrics": [m[0] for m in metrics], "source_keys": ["last_summary_metrics", "best_summary_metrics"], "statistics": "mean with 95% block-20 interval; common paper edges and 66x66 overflow edges", "coordinates": records, "visual_encoding": "circle/policy points for paper JSD; x markers for overflow JSD; A0 red/A1 slate", "axes": {"x": "JSD base 2", "pair_targets": PAIR_TARGETS}})
    return fig


def make_entry_composite(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, output_base: Path, si_key: str) -> None:
    # Reuse the full plotting functions on nested SubFigures.  This keeps the
    # entry-point artwork vector/editable while preserving every policy,
    # interval, method label and metric shown by its standalone counterpart.
    groups = {
        # The entry-point SI01 composite is deliberately focused on the two
        # requested reconstruction panels.  Normalisation checks and the
        # CH4 local-conditioning detail remain available as authoritative
        # standalone subpanels and in the accompanying tables.
        "si01": ["si01a_fieldwise_physical_l2", "si01b_fieldwise_effect_heatmap"],
        "si02": ["si02a_all_field_highband_l2", "si02b_all_field_canonical_power"],
        "si03": ["si03a_u1_population_spectra", "si03b_u1_mode_sum_ecdf", "si03c_u1_hann_sensitivity"],
        "si04": ["si04_joint_pdf_jsd_base2", "si04_joint_pdf_jsd_with_overflow_base2", "si04c_coupling_retention"],
        "si05": ["si05a_checkpoint_sensitivity", "si05b_block_length_sensitivity", "si05c_recorded_training_histories"],
        "si06": ["si06a_deterministic_macro_control", "si06b_deterministic_fieldwise_control", "si06c_deterministic_coupling_control"],
    }
    titles = {
        "si01a_fieldwise_physical_l2": "Fieldwise physical reconstruction error",
        "si01b_fieldwise_effect_heatmap": "Fieldwise effect relative to full",
        "si01c_normalization_checks": "Standardized and truth-fluctuation normalizations",
        "si01d_local_conditioning_ch4_detail": "Local-conditioning CH$_4$ detail",
        "si02a_all_field_highband_l2": "All-field high-band residual",
        "si02b_all_field_canonical_power": "All-field canonical high-band power",
        "si03a_u1_population_spectra": "U1 population shell spectra",
        "si03b_u1_mode_sum_ecdf": "Mode-sum power and residual ECDF",
        "si03c_u1_hann_sensitivity": "Hann-window sensitivity",
        "si04_joint_pdf_jsd_base2": "Paper-range coupling JSD",
        "si04_joint_pdf_jsd_with_overflow_base2": "Overflow-retaining coupling JSD",
        "si04c_coupling_retention": "Coupled-pair retention",
        "si05a_checkpoint_sensitivity": "Best-minus-last checkpoint sensitivity",
        "si05b_block_length_sensitivity": "Block-length sensitivity",
        "si05c_recorded_training_histories": "Recorded training and validation histories",
        "si06a_deterministic_macro_control": "Deterministic objective control",
        "si06b_deterministic_fieldwise_control": "Deterministic fieldwise control",
        "si06c_deterministic_coupling_control": "Deterministic coupling control",
    }

    class _NoopManifest:
        def add_plot(self, _record: dict[str, Any]) -> None:
            return None

    noop = _NoopManifest()
    draw_map: dict[str, Any] = {
        "si01a_fieldwise_physical_l2": draw_si01_fieldwise,
        "si01b_fieldwise_effect_heatmap": draw_si01_heatmap,
        "si01c_normalization_checks": draw_si01_normalizations,
        "si01d_local_conditioning_ch4_detail": draw_si01_local_detail,
        "si02a_all_field_highband_l2": draw_si02_highband_l2,
        "si02b_all_field_canonical_power": draw_si02_power,
        "si03a_u1_population_spectra": draw_si03_spectra,
        "si03b_u1_mode_sum_ecdf": draw_si03_ecdf,
        "si03c_u1_hann_sensitivity": draw_si03_hann,
        "si04c_coupling_retention": draw_si04_retention,
        "si05a_checkpoint_sensitivity": draw_si05_checkpoint,
        "si05b_block_length_sensitivity": draw_si05_blocks,
        "si05c_recorded_training_histories": draw_si05_histories,
        "si06a_deterministic_macro_control": draw_si06_macro,
        "si06b_deterministic_fieldwise_control": draw_si06_fields,
        "si06c_deterministic_coupling_control": draw_si06_coupling,
    }
    heights = {
        "si01": [2.85, 2.75],
        "si02": [2.9, 2.9],
        "si03": [2.65, 3.0, 2.65],
        "si04": [2.65, 2.65, 2.65],
        "si05": [2.65, 2.65, 2.65],
        "si06": [2.65, 2.65, 2.65],
    }
    fig = plt.figure(figsize=(183 / 25.4, sum(heights[si_key]) * 25.4 / 25.4))
    outer = fig.add_gridspec(len(groups[si_key]), 1, height_ratios=heights[si_key], hspace=0.42)
    for idx, panel_id in enumerate(groups[si_key]):
        subfig = fig.add_subfigure(outer[idx, 0])
        subfig.text(0.01, 0.99, titles[panel_id], ha="left", va="top", fontsize=6.6, fontweight="semibold")
        # Keep a right-hand gutter inside the fixed 183 mm canvas for the
        # longer method legends used by the coupling and block panels.
        subfig.subplots_adjust(left=0.18, right=0.78, bottom=0.20, top=0.88, wspace=0.34, hspace=0.45)
        if panel_id == "si04_joint_pdf_jsd_base2":
            subaxes = subfig.subplots(1, 2, sharey=True)
            draw_si04_metric(data, config, noop, "joint_pdf_jsd_base2", "Paper-range JSD (base 2)", "a", axes=subaxes, figure=subfig)
        elif panel_id == "si04_joint_pdf_jsd_with_overflow_base2":
            subaxes = subfig.subplots(1, 2, sharey=True)
            draw_si04_metric(data, config, noop, "joint_pdf_jsd_with_overflow_base2", "Overflow-retaining JSD (base 2)", "b", axes=subaxes, figure=subfig)
        else:
            fn = draw_map[panel_id]
            if panel_id == "si01c_normalization_checks":
                subaxes = subfig.subplots(2, 2, sharey="row")
            elif panel_id == "si03b_u1_mode_sum_ecdf":
                subaxes = subfig.subplots(2, 2, sharex="col", sharey="row")
            elif panel_id == "si01d_local_conditioning_ch4_detail":
                subaxes = [subfig.subplots(1, 1)]
            elif panel_id in {"si05b_block_length_sensitivity", "si05c_recorded_training_histories"}:
                # These columns represent different metrics or losses, so
                # each must retain its standalone vertical scale.
                subaxes = subfig.subplots(1, 2, sharey=False)
            else:
                subaxes = subfig.subplots(1, 2, sharey=True)
            fn(data, config, noop, chr(ord("a") + idx), axes=subaxes, figure=subfig)
    manifest.add_plot({
        "id": si_key,
        "role": f"SI entry-point composite {si_key}",
        "linked_subpanels": groups[si_key],
        "layout": {"rows": len(groups[si_key]), "columns": 1, "height_ratios_in": heights[si_key], "standalone_authoritative": True},
        "content_policy": "full nested subpanels retain both last.pt and best.pt policies, source intervals, all requested methods and metric-specific guides",
        "visual_encoding": "stacked full-width vector subpanels; standalone and entry-point coordinates are source-identical",
    })
    # Entry composites retain the configured 183 mm canvas width.  Standalone
    # panels use a tight crop; composites reserve their full-width layout so
    # the exported physical size remains stable for LaTeX insertion.
    save_figure(fig, output_base, dpi=int(config.get("figure", {}).get("png_dpi", 600)), qa_sink=manifest.qa.setdefault("visual", []), crop=False)


def render_all(data: SourceBundle, config: Mapping[str, Any], manifest: PlotManifest, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    standalone = [
        ("si01a_fieldwise_physical_l2", draw_si01_fieldwise, "a"),
        ("si01b_fieldwise_effect_heatmap", draw_si01_heatmap, "b"),
        ("si01c_normalization_checks", draw_si01_normalizations, "c"),
        ("si01d_local_conditioning_ch4_detail", draw_si01_local_detail, "d"),
        ("si02a_all_field_highband_l2", draw_si02_highband_l2, "a"),
        ("si02b_all_field_canonical_power", draw_si02_power, "b"),
        ("si03a_u1_population_spectra", draw_si03_spectra, "a"),
        ("si03b_u1_mode_sum_ecdf", draw_si03_ecdf, "b"),
        ("si03c_u1_hann_sensitivity", draw_si03_hann, "c"),
        ("si04_joint_pdf_jsd_base2", lambda d, c, m, t: draw_si04_metric(d, c, m, "joint_pdf_jsd_base2", "Paper-range JSD (base 2)", t), "a"),
        ("si04_joint_pdf_jsd_with_overflow_base2", lambda d, c, m, t: draw_si04_metric(d, c, m, "joint_pdf_jsd_with_overflow_base2", "Overflow-retaining JSD (base 2)", t), "b"),
        ("si04c_coupling_retention", draw_si04_retention, "c"),
        ("si05a_checkpoint_sensitivity", draw_si05_checkpoint, "a"),
        ("si05b_block_length_sensitivity", draw_si05_blocks, "b"),
        ("si05c_recorded_training_histories", draw_si05_histories, "c"),
        ("si06a_deterministic_macro_control", draw_si06_macro, "a"),
        ("si06b_deterministic_fieldwise_control", draw_si06_fields, "b"),
        ("si06c_deterministic_coupling_control", draw_si06_coupling, "c"),
    ]
    for panel_id, fn, tag in standalone:
        fig = fn(data, config, manifest, tag)
        generated.extend(save_figure(fig, output_dir / f"{panel_id}_{manifest.timestamp}", dpi=int(config.get("figure", {}).get("png_dpi", 600)), qa_sink=manifest.qa.setdefault("visual", [])))

    entry_names = {
        "si01": "si_ablation_reconstruction",
        "si02": "si_ablation_highband_all_fields",
        "si03": "si_ablation_u1_spectral_audit",
        "si04": "si_ablation_coupling_range_audit",
        "si05": "si_ablation_checkpoint_sensitivity",
        "si06": "si_deterministic_objective_control",
    }
    for si_key, stem in entry_names.items():
        before = len(generated)
        make_entry_composite(data, config, manifest, output_dir / f"{stem}_{manifest.timestamp}", si_key)
        generated.extend([relpath(output_dir / f"{stem}_{manifest.timestamp}.svg"), relpath(output_dir / f"{stem}_{manifest.timestamp}.png")])
    return generated


def build_manifest_scope(config: Mapping[str, Any], data: SourceBundle) -> dict[str, Any]:
    return {
        "type": "saved_checkpoint_comparison",
        "matched_evaluation_states": True,
        "matched_measurements": True,
        "matched_training_budget": False,
        "multiple_training_seeds": False,
        "independent_untouched_test_set": False,
        "matched_backbone_initialization_verified": False,
        "generative_draws_per_ablation_state": 1,
        "primary_checkpoint_policy": "last",
        "sensitivity_checkpoint_policy": "best",
        "new_ablation_reference_separate_from_benchmark": True,
        "deterministic_objective_control_retained_in_si": True,
        "states_per_method_policy": 1000,
        "measurements": 256,
        "queries": 40300,
        "sampler": "one draw; two Euler steps; recorded hard observed-entry clamp",
        "fields": FIELDS,
        "primary_unobserved_fields": UNOBSERVED_FIELDS,
        "bootstrap": {"kind": "circular moving-block", "replicates": 2000, "blocks": [5, 20, 50], "source_seed": 20260906, "high_frequency_source_seed": 20260910},
        "spectral_definition": {"coordinate_mode": "index-space/topological", "grid": [403, 100], "high_band_rule": "k > (2/3)*k_max", "high_shell_count": 68, "high_mode_count": 17704, "canonical_power": "shell-mean trapezoidal integral", "highband_residual": "individual complex FFT coefficient mode sum", "hann_separate": True},
        "coupling_definition": {"base": 2, "paper_edges": "64x64 common archived truth edges", "overflow_edges": "66x66 with under/overflow bins", "retention_reported": True},
        "history_policy": "raw recorded train/validation rows; no smoothing or endpoint reconstruction",
    }


def interval_clip_qa(manifest: PlotManifest) -> dict[str, Any]:
    """Audit every displayed CI endpoint against its declared x-axis range."""
    clipped: list[dict[str, Any]] = []
    for plot in manifest.plots:
        axes = plot.get("axes", {}) if isinstance(plot, Mapping) else {}
        xlim = axes.get("xlim") if isinstance(axes, Mapping) else None
        if not (isinstance(xlim, (list, tuple)) and len(xlim) == 2):
            continue
        try:
            lo_axis, hi_axis = float(xlim[0]), float(xlim[1])
        except (TypeError, ValueError):
            continue
        coords = plot.get("coordinates", [])
        if not isinstance(coords, list):
            continue
        for coord in coords:
            ci = coord.get("ci95_block20") if isinstance(coord, Mapping) else None
            if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
                continue
            try:
                low, high = float(ci[0]), float(ci[1])
            except (TypeError, ValueError):
                continue
            if low < lo_axis - 1e-9 or high > hi_axis + 1e-9:
                clipped.append({"plot_id": plot.get("id"), "method_key": coord.get("method_key"), "metric": coord.get("metric"), "target": coord.get("target"), "ci95_block20": [low, high], "xlim": [lo_axis, hi_axis]})
    visual = manifest.qa.get("visual", []) if isinstance(manifest.qa, Mapping) else []
    off_canvas = [record for record in visual if record.get("off_canvas_text")]
    return {"figure_count": len(visual), "off_canvas_text": off_canvas, "clipped_intervals": clipped, "status": "pass" if not off_canvas and not clipped else "fail"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--strict-formal", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(config_path)
    apply_style(config)
    evaluation_root = args.evaluation_dir or Path(str(config.get("evaluation_root", DEFAULT_EVAL)))
    if not evaluation_root.is_absolute():
        evaluation_root = REPO_ROOT / evaluation_root
    output_root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    figure_dir = output_root / "Dis_SI_Process" / "figures" / "generated" / args.timestamp / "si"
    manifest = PlotManifest(
        timestamp=args.timestamp,
        renderer=relpath(SCRIPT_PATH),
        config_path=relpath(config_path),
        evaluation_root=relpath(evaluation_root),
        scope=build_manifest_scope(config, SourceBundle.__new__(SourceBundle)),
    )
    # Replace the temporary scope placeholder with a source-independent scope;
    # keeping this separate prevents any accidental source mutation.
    manifest.scope = build_manifest_scope(config, SourceBundle.__new__(SourceBundle))
    try:
        # Freeze live training histories before loading any source tables.  A
        # rerun at this timestamp validates and reuses the same compact bytes,
        # so a later optimizer append cannot alter the SI artwork or hashes.
        capture_history_snapshot(evaluation_root, figure_dir, args.timestamp)
        data = load_sources(evaluation_root, manifest, history_snapshot_dir=figure_dir)
        manifest.scope = build_manifest_scope(config, data)
        qa = validate_sources(data, args.strict_formal)
        manifest.qa = qa
        render_all(data, config, manifest, figure_dir)
        manifest.qa["visual_summary"] = interval_clip_qa(manifest)
        if args.strict_formal and manifest.qa["visual_summary"]["status"] != "pass":
            raise RuntimeError("strict SI visual validation failed: off-canvas text or clipped confidence interval detected")
    except Exception as exc:
        manifest.qa = {"status": "blocked", "error": str(exc)}
        figure_dir.mkdir(parents=True, exist_ok=True)
        manifest.write(figure_dir / "si_plot_manifest.json")
        raise
    manifest.write(figure_dir / "si_plot_manifest.json")
    print(json.dumps({"status": "pass", "figure_dir": relpath(figure_dir), "manifest": relpath(figure_dir / "si_plot_manifest.json"), "plots": len(manifest.plots)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

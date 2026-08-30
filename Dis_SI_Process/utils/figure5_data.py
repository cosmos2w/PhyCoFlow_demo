"""Strict adapters from frozen ValidationV2 products to Figure 5 V2 panels."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class SourceRecord:
    panel: str
    mode: str
    status: str
    source: str
    note: str


def _repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_runs(root: Path, *, kind: str, job: str | None = None) -> list[Path]:
    candidates = []
    if not root.exists():
        return candidates
    for directory in root.iterdir():
        manifest_path, qa_path = directory / "manifest.json", directory / "qa.json"
        if not directory.is_dir() or not manifest_path.is_file() or not qa_path.is_file():
            continue
        try:
            manifest, qa = _json(manifest_path), _json(qa_path)
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "complete" or qa.get("status") != "pass" or not manifest.get("formal", False):
            continue
        if kind == "uncertainty" and manifest.get("job") != job:
            continue
        candidates.append(directory)
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _load_uncertainty(root: Path, job: str, *, expected_states: int, expected_draws: int) -> dict[str, Any] | None:
    for directory in _candidate_runs(root, kind="uncertainty", job=job):
        per_state_path, coverage_path = directory / "per_state_field.csv", directory / "coverage_by_level.csv"
        if not per_state_path.is_file() or not coverage_path.is_file():
            continue
        per_state, coverage = pd.read_csv(per_state_path), pd.read_csv(coverage_path)
        if len(per_state) != expected_states * 5:
            continue
        if per_state["state"].nunique() != expected_states or set(per_state["draw_count"].astype(int)) != {expected_draws}:
            continue
        if per_state.select_dtypes(include=[np.number]).isna().any().any():
            continue
        return {"directory": directory, "manifest": _json(directory / "manifest.json"), "qa": _json(directory / "qa.json"), "per_state": per_state, "coverage": coverage}
    return None


def _load_cost(root: Path, methods: list[str]) -> dict[str, Any] | None:
    for directory in _candidate_runs(root, kind="cost"):
        summary_path, nfe_path = directory / "benchmark_summary.csv", directory / "nfe_error.csv"
        if not summary_path.is_file() or not nfe_path.is_file():
            continue
        summary, nfe = pd.read_csv(summary_path), pd.read_csv(nfe_path)
        timed = summary[summary["status"].eq("ok")]
        if "timed_total_ms" not in timed.columns or (timed["repeats"].fillna(0) < 30).any() or (timed["timed_total_ms"].fillna(0) < 10000.0).any():
            continue
        native = summary[summary["suite"].eq("native_methods")].copy()
        query = summary[summary["suite"].eq("dmf_query_memory") & summary["status"].eq("ok")].copy()
        statuses = dict(zip(native["method"].astype(str), native["status"].astype(str)))
        if set(statuses) != set(methods):
            continue
        if set(query["N"].dropna().astype(int)) != {1024, 4096, 16384, 40300}:
            continue
        if set(nfe["measured_nfe"].astype(int)) != {1, 2, 4, 8}:
            continue
        return {"directory": directory, "manifest": _json(directory / "manifest.json"), "qa": _json(directory / "qa.json"), "native": native, "query": query, "nfe": nfe}
    return None


def _bin_spread_error(table: pd.DataFrame, field_order: list[str], bins: int, qa: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    associations = {row["field"]: row for row in qa.get("associations", [])}
    for field in field_order:
        group = table[table["field"].eq(field)].sort_values("original_time_index").copy()
        spread = group["spread_rms_normalized"].to_numpy(dtype=float)
        error = group["ensemble_mean_relative_l2"].to_numpy(dtype=float)
        edges = np.unique(np.quantile(spread, np.linspace(0, 1, bins + 1)))
        group["bin"] = np.clip(np.digitize(spread, edges[1:-1]), 0, max(len(edges) - 2, 0))
        for bin_id, part in group.groupby("bin", sort=True):
            records.append({"field": field, "bin": int(bin_id), "spread": float(part["spread_rms_normalized"].mean()), "error": float(part["ensemble_mean_relative_l2"].mean()), "error_q25": float(part["ensemble_mean_relative_l2"].quantile(0.25)), "error_q75": float(part["ensemble_mean_relative_l2"].quantile(0.75)), "n": int(len(part))})
        if field not in associations:
            associations[field] = {"field": field, "spearman_rho": float(spearmanr(spread, error).statistic), "spearman_ci_low": np.nan, "spearman_ci_high": np.nan}
    return {"table": pd.DataFrame(records), "associations": associations, "raw": table}


def load_figure5_data(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[SourceRecord]]:
    """Load formal products only; absent evidence remains pending, never proxied."""
    fields = list(config["paper_contract"]["unobserved_fields"])
    methods = list(config["paper_contract"]["method_order"])
    uq_root = _repo_path(repo_root, config["formal_inputs"]["uncertainty_root"])
    cost_root = _repo_path(repo_root, config["formal_inputs"]["cost_root"])
    u1 = _load_uncertainty(uq_root, "U1", expected_states=1000, expected_draws=16)
    u2 = _load_uncertainty(uq_root, "U2", expected_states=200, expected_draws=64)
    cost = _load_cost(cost_root, methods)
    modes = {"a": "formal" if u2 else "pending", "b": "formal" if u2 else "pending", "c": "formal" if u1 else "pending", "d": "formal" if cost else "pending", "e": "formal" if cost else "pending", "f": "formal" if cost else "pending"}
    sources = {
        "a": str(u2["directory"] / "coverage_by_level.csv") if u2 else str(uq_root / "<formal-U2>"),
        "b": str(u2["directory"] / "coverage_by_level.csv") if u2 else str(uq_root / "<formal-U2>"),
        "c": str(u1["directory"] / "per_state_field.csv") if u1 else str(uq_root / "<formal-U1>"),
        "d": str(cost["directory"] / "benchmark_summary.csv") if cost else str(cost_root / "<formal-cost>"),
        "e": str(cost["directory"] / "benchmark_summary.csv") if cost else str(cost_root / "<formal-cost>"),
        "f": str(cost["directory"] / "nfe_error.csv") if cost else str(cost_root / "<formal-cost>"),
    }
    data: dict[str, Any] = {"modes": modes, "sources": sources, "coverage": None, "spread_error": None, "cost_native": None, "cost_query": None, "cost_nfe": None, "run_metadata": {"U1": u1, "U2": u2, "cost": cost}}
    if u2:
        coverage = u2["coverage"].copy()
        data["coverage"] = coverage[coverage["field"].isin(fields) & coverage["sensor_count"].eq(256)].copy()
    if u1:
        per_state = u1["per_state"]
        per_state = per_state[per_state["field"].isin(fields) & per_state["sensor_count"].eq(256)].copy()
        data["spread_error"] = _bin_spread_error(per_state, fields, int(config["formal_protocol"]["uq_spread_error"]["bins"]), u1["qa"])
    if cost:
        native = cost["native"].copy()
        native["method"] = pd.Categorical(native["method"], categories=methods, ordered=True)
        data["cost_native"] = native.sort_values("method")
        data["cost_query"] = cost["query"].sort_values("N")
        data["cost_nfe"] = cost["nfe"].sort_values("measured_nfe")
    notes = {
        "a": "Formal U2 state-level empirical coverage." if u2 else "Requires a complete formal U2 run (200 states × 64 draws).",
        "b": "Formal U2 widths normalized by frozen training standard deviations." if u2 else "Requires the same complete formal U2 run.",
        "c": "Formal U1 state-level spread/error association." if u1 else "Requires a complete formal U1 run (1,000 states × 16 draws).",
        "d": "Formal canonical-adapter native benchmark with frozen FieldL2 join." if cost else "Requires a complete formal all-suite cost run.",
        "e": "Formal DMF real-coordinate query/memory sweep." if cost else "Requires N=1,024/4,096/16,384/40,300 benchmark rows.",
        "f": "Formal DMF measured-NFE accuracy/latency sweep." if cost else "Requires the fixed 50-state NFE sweep.",
    }
    records = [SourceRecord(panel=panel, mode=modes[panel], status="available" if modes[panel] == "formal" else "missing", source=sources[panel], note=notes[panel]) for panel in "abcdef"]
    return data, records

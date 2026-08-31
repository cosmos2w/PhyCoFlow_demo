"""Strict adapters from frozen ValidationV3 products to Figure 5 V3."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _finite(table: pd.DataFrame, columns: list[str]) -> bool:
    return bool(np.isfinite(table[columns].to_numpy(dtype=float)).all())


def _formal_run(root: Path, run_id: str, schema: str, required: list[str]) -> dict[str, Any] | None:
    directory = root / run_id
    manifest_path, qa_path = directory / "manifest.json", directory / "qa.json"
    if not manifest_path.is_file() or not qa_path.is_file():
        return None
    manifest, qa = _json(manifest_path), _json(qa_path)
    if manifest.get("schema_version") != schema or manifest.get("status") != "complete" or not manifest.get("formal", False) or qa.get("status") != "pass":
        return None
    if not all((directory / name).is_file() for name in required):
        return None
    return {"directory": directory, "manifest": manifest, "qa": qa}


def _load_uq(config: dict[str, Any], repo_root: Path) -> dict[str, Any] | None:
    formal = config["formal_inputs"]
    root = _repo_path(repo_root, formal["uq_root"])
    run = _formal_run(root, formal["uq_run_id"], "figure5-validation-v3-uq-1", ["crps_summary.csv", "spread_error_summary.csv", "per_state_method.csv", "reliability_si.csv"])
    if run is None or "ValidationV2" in str(run["directory"]):
        return None
    crps = pd.read_csv(run["directory"] / "crps_summary.csv")
    spread = pd.read_csv(run["directory"] / "spread_error_summary.csv")
    states = pd.read_csv(run["directory"] / "per_state_method.csv")
    methods = config["paper_contract"]["generative_method_order"]
    if list(crps["method"].astype(str)) != methods or list(spread["method"].astype(str)) != methods:
        return None
    if len(states) != 1000 or states["state"].nunique() != 200 or set(states["draw_count"].astype(int)) != {64}:
        return None
    if any(states[states["method"].eq(method)]["state"].nunique() != 200 for method in methods):
        return None
    if not _finite(crps, ["mean_normalized_crps", "crps_ci_low", "crps_ci_high"]) or not _finite(spread, ["spearman_rho", "spearman_ci_low", "spearman_ci_high"]):
        return None
    run.update({"crps": crps, "spread": spread, "states": states})
    return run


def _load_cost(config: dict[str, Any], repo_root: Path) -> dict[str, Any] | None:
    formal = config["formal_inputs"]
    root = _repo_path(repo_root, formal["cost_root"])
    run = _formal_run(root, formal["cost_run_id"], "figure5-validation-v3-cost-1", ["native_summary.csv", "query_latency_summary.csv", "memory_summary.csv", "variable_query_support.csv", "timing_boundary_audit.csv"])
    if run is None or "ValidationV2" in str(run["directory"]):
        return None
    native = pd.read_csv(run["directory"] / "native_summary.csv")
    query = pd.read_csv(run["directory"] / "query_latency_summary.csv")
    memory = pd.read_csv(run["directory"] / "memory_summary.csv")
    support = pd.read_csv(run["directory"] / "variable_query_support.csv")
    boundary = pd.read_csv(run["directory"] / "timing_boundary_audit.csv")
    methods = config["paper_contract"]["method_order"]
    if list(native["method"].astype(str)) != methods or set(native["status"].astype(str)) != {"ok"}:
        return None
    if set(support["method"].astype(str)) != set(methods):
        return None
    variable = set(support[support["variable_query_supported"].astype(bool)]["method"].astype(str))
    expected = {(method, count) for method in methods for count in ([1024, 4096, 16384, 40300] if method in variable else [40300])}
    query_keys = set(zip(query["method"].astype(str), query["N"].astype(int)))
    memory_keys = set(zip(memory["method"].astype(str), memory["N"].astype(int)))
    if query_keys != expected or memory_keys != expected or query_keys != memory_keys:
        return None
    if not _finite(native, ["median_latency_ms", "latency_q25_ms", "latency_q75_ms", "error", "error_ci_low", "error_ci_high"]):
        return None
    if not _finite(query, ["median_latency_ms", "latency_q25_ms", "latency_q75_ms"]) or not _finite(memory, ["peak_allocated_mib"]):
        return None
    if len(boundary) < 3 or not boundary.iloc[:3]["pass_20pct"].astype(bool).all():
        return None
    run.update({"native": native, "query": query, "memory": memory, "support": support, "boundary": boundary})
    return run


def load_figure5_data(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[SourceRecord]]:
    """Load only adopted V3 formal products; missing inputs remain pending."""
    uq = _load_uq(config, repo_root)
    cost = _load_cost(config, repo_root)
    modes = {"a": "formal" if uq else "pending", "b": "formal" if uq else "pending", "c": "formal" if cost else "pending", "d": "formal" if cost else "pending", "e": "formal" if cost else "pending"}
    uq_root = _repo_path(repo_root, config["formal_inputs"]["uq_root"]) / config["formal_inputs"]["uq_run_id"]
    cost_root = _repo_path(repo_root, config["formal_inputs"]["cost_root"]) / config["formal_inputs"]["cost_run_id"]
    sources = {
        "a": str((uq["directory"] if uq else uq_root) / "crps_summary.csv"),
        "b": str((uq["directory"] if uq else uq_root) / "spread_error_summary.csv"),
        "c": str((cost["directory"] if cost else cost_root) / "native_summary.csv"),
        "d": str((cost["directory"] if cost else cost_root) / "query_latency_summary.csv"),
        "e": str((cost["directory"] if cost else cost_root) / "memory_summary.csv"),
    }
    notes = {
        "a": "Five-method paired normalized empirical CRPS with equal field weights." if uq else "Requires the complete five-method 200-state x 64-draw V3 UQ run.",
        "b": "Five method-wise macro spread/error Spearman estimates." if uq else "Requires complete cross-model V3 UQ summaries.",
        "c": "Eight exact checkpoints with clean-GPU model-core timing and frozen FieldL2." if cost else "Requires a passing clean V3 native benchmark and DMF reconciliation.",
        "d": "Curves only for audited native variable-query models; fixed-grid methods are native-only." if cost else "Requires the V3 query-support audit and latency table.",
        "e": "Peak allocated memory under the identical support/query protocol as panel d." if cost else "Requires the V3 memory table with matching support keys.",
    }
    records = [SourceRecord(panel=panel, mode=modes[panel], status="available" if modes[panel] == "formal" else "missing", source=sources[panel], note=notes[panel]) for panel in "abcde"]
    return {
        "modes": modes,
        "sources": sources,
        "uq_crps": None if uq is None else uq["crps"],
        "uq_spread": None if uq is None else uq["spread"],
        "cost_native": None if cost is None else cost["native"],
        "cost_query": None if cost is None else cost["query"],
        "cost_memory": None if cost is None else cost["memory"],
        "query_support": None if cost is None else cost["support"],
        "timing_boundary": None if cost is None else cost["boundary"],
        "run_metadata": {"uq": uq, "cost": cost},
    }, records

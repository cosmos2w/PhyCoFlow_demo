"""Adapters from existing/future result products to the Figure 5 panel contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


def _latest(paths: Iterable[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None


def _column(df: pd.DataFrame, *names: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _grid(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.unique(coords[:, 0])
    y = np.unique(coords[:, 1])
    out = np.full((len(y), len(x)), np.nan, dtype=float)
    xi = np.searchsorted(x, coords[:, 0])
    yi = np.searchsorted(y, coords[:, 1])
    out[yi, xi] = values
    return x, y, out


def _load_npz_map(path: Path, field: str) -> dict[str, Any] | None:
    z = np.load(path, allow_pickle=False)
    keys = set(z.files)
    coord_key = next((k for k in ("coords_raw", "coords", "query_coords") if k in keys), None)
    truth_key = next((k for k in ("truth_phys", "truth", "target") if k in keys), None)
    mean_key = next((k for k in ("ensemble_mean", "mean", "recon_phys", "reconstruction") if k in keys), None)
    std_key = next((k for k in ("ensemble_std", "std", "predictive_std") if k in keys), None)
    field_key = next((k for k in ("field_names", "fields", "channels") if k in keys), None)
    if not coord_key or not truth_key or not mean_key:
        return None
    names = [str(x) for x in z[field_key]] if field_key else [str(i) for i in range(z[truth_key].shape[-1])]
    if field not in names:
        return None
    idx = names.index(field)
    coords = np.asarray(z[coord_key])
    truth = np.asarray(z[truth_key])[:, idx]
    mean = np.asarray(z[mean_key])[:, idx]
    std = np.asarray(z[std_key])[:, idx] if std_key else np.full_like(truth, np.nan)
    x, y, truth_grid = _grid(coords, truth)
    _, _, mean_grid = _grid(coords, mean)
    _, _, std_grid = _grid(coords, std)
    return {
        "x": x,
        "y": y,
        "truth": truth_grid,
        "mean": mean_grid,
        "error": np.abs(mean_grid - truth_grid),
        "std": std_grid,
        "field": field,
        "relative_l2": float(np.linalg.norm(mean - truth) / max(np.linalg.norm(truth), 1e-12)),
    }


def _load_formal_uq(config: dict[str, Any], repo_root: Path, field: str) -> dict[str, Any]:
    root = _repo_path(repo_root, config["formal_inputs"]["uncertainty_root"])
    result: dict[str, Any] = {"root": root, "map": None, "coverage": None, "spread_error": None}
    map_path = _latest(root.glob("*/visual_maps/*.npz")) if root.exists() else None
    if map_path:
        result["map"] = _load_npz_map(map_path, field)
        result["map_source"] = map_path

    coverage_path = _latest(root.glob("*/coverage_by_level.csv")) if root.exists() else None
    if coverage_path:
        raw = pd.read_csv(coverage_path)
        fcol = _column(raw, "field", "channel")
        nominal = _column(raw, "nominal_level", "level", "nominal_coverage")
        empirical = _column(raw, "empirical_coverage", "coverage", "mean_coverage")
        width = _column(raw, "mean_interval_width", "interval_width", "width")
        if fcol and nominal and empirical:
            keep = pd.DataFrame(
                {
                    "field": raw[fcol].astype(str),
                    "nominal": pd.to_numeric(raw[nominal], errors="coerce"),
                    "empirical": pd.to_numeric(raw[empirical], errors="coerce"),
                    "width": pd.to_numeric(raw[width], errors="coerce") if width else np.nan,
                }
            ).dropna(subset=["nominal", "empirical"])
            if keep["nominal"].max() > 1.5:
                keep[["nominal", "empirical"]] /= 100.0
            result["coverage"] = keep
            result["coverage_source"] = coverage_path

    state_path = _latest(root.glob("*/per_state_field.csv")) if root.exists() else None
    if state_path:
        raw = pd.read_csv(state_path)
        fcol = _column(raw, "field", "channel")
        spread = _column(raw, "spread_rms", "spatial_rms_std", "ensemble_spread")
        error = _column(raw, "ensemble_mean_relative_l2", "ensemble_mean_rel_l2", "relative_l2", "error")
        if fcol and spread and error:
            keep = pd.DataFrame(
                {
                    "field": raw[fcol].astype(str),
                    "spread": pd.to_numeric(raw[spread], errors="coerce"),
                    "error": pd.to_numeric(raw[error], errors="coerce"),
                }
            ).dropna()
            result["spread_error"] = _bin_spread_error(keep, 8)
            result["spread_error_source"] = state_path
    return result


def _bin_spread_error(values: pd.DataFrame, bins: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    rho: dict[str, float] = {}
    for field, group in values.groupby("field", sort=False):
        group = group[np.isfinite(group["spread"]) & np.isfinite(group["error"])].copy()
        if len(group) < 3:
            continue
        rho[field] = float(spearmanr(group["spread"], group["error"]).statistic)
        edges = np.unique(np.quantile(group["spread"], np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        group["bin"] = np.clip(np.digitize(group["spread"], edges[1:-1]), 0, len(edges) - 2)
        for bin_id, part in group.groupby("bin"):
            records.append(
                {
                    "field": field,
                    "bin": int(bin_id),
                    "spread": float(part["spread"].mean()),
                    "error": float(part["error"].mean()),
                    "error_q25": float(part["error"].quantile(0.25)),
                    "error_q75": float(part["error"].quantile(0.75)),
                    "n": int(len(part)),
                }
            )
    return {"table": pd.DataFrame.from_records(records), "rho": rho}


def _load_draft_uq(config: dict[str, Any], repo_root: Path, field: str) -> dict[str, Any]:
    pattern = config["draft_inputs"]["qualitative_nfe_pattern"]
    nfes = [int(x) for x in config["draft_inputs"]["qualitative_nfes"]]
    paths = [_repo_path(repo_root, pattern.format(nfe=nfe)) for nfe in nfes]
    arrays = [np.load(path, allow_pickle=False) for path in paths]
    names = [str(x) for x in arrays[0]["field_names"]]
    if field not in names:
        field = names[0]
    idx = names.index(field)
    coords = arrays[0]["coords_raw"]
    truth = arrays[0]["truth_phys"][:, idx]
    middle_index = nfes.index(2) if 2 in nfes else len(nfes) // 2
    mean = arrays[middle_index]["recon_phys"][:, idx]
    stack = np.stack([z["recon_phys"][:, idx] for z in arrays])
    sensitivity = np.std(stack, axis=0, ddof=1)
    x, y, truth_grid = _grid(coords, truth)
    _, _, mean_grid = _grid(coords, mean)
    _, _, sensitivity_grid = _grid(coords, sensitivity)
    map_data = {
        "x": x,
        "y": y,
        "truth": truth_grid,
        "mean": mean_grid,
        "error": np.abs(mean_grid - truth_grid),
        "std": sensitivity_grid,
        "field": field,
        "relative_l2": float(np.linalg.norm(mean - truth) / max(np.linalg.norm(truth), 1e-12)),
    }

    unobserved = [x for x in config["statistics"]["unobserved_fields"] if x in names]
    proxy_rows = []
    for name in unobserved:
        j = names.index(name)
        truth_j = arrays[0]["truth_phys"][:, j]
        center_j = arrays[middle_index]["recon_phys"][:, j]
        spread_j = np.std(np.stack([z["recon_phys"][:, j] for z in arrays]), axis=0, ddof=1)
        scale = max(float(np.sqrt(np.mean(truth_j**2))), 1e-12)
        for spread_value, error_value in zip(spread_j / scale, np.abs(center_j - truth_j) / scale):
            proxy_rows.append({"field": name, "spread": spread_value, "error": error_value})
    proxy = _bin_spread_error(pd.DataFrame(proxy_rows), int(config["statistics"]["spread_error_bins"]))
    return {"map": map_data, "spread_error": proxy, "map_source": paths[middle_index], "proxy_sources": paths}


def _load_formal_cost(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    root = _repo_path(repo_root, config["formal_inputs"]["cost_root"])
    result: dict[str, Any] = {"root": root, "native": None, "query": None, "nfe": None}
    summary_path = _latest(root.glob("*/benchmark_summary.csv")) if root.exists() else None
    if summary_path:
        raw = pd.read_csv(summary_path)
        method = _column(raw, "method", "model")
        suite = _column(raw, "suite", "benchmark")
        n_query = _column(raw, "N", "N_query", "query_count")
        latency = _column(raw, "median_latency_ms", "latency_ms_median", "warm_latency_median_ms", "latency_ms")
        memory = _column(raw, "peak_allocated_mib", "peak_allocated_mb", "peak_memory_mib")
        error = _column(raw, "unobserved_mean_error", "error", "relative_l2")
        normalized = pd.DataFrame(index=raw.index)
        normalized["method"] = raw[method].astype(str) if method else "DMF-Gen"
        normalized["suite"] = raw[suite].astype(str) if suite else ""
        normalized["N"] = pd.to_numeric(raw[n_query], errors="coerce") if n_query else np.nan
        normalized["latency_ms"] = pd.to_numeric(raw[latency], errors="coerce") if latency else np.nan
        normalized["memory_mib"] = pd.to_numeric(raw[memory], errors="coerce") if memory else np.nan
        normalized["error"] = pd.to_numeric(raw[error], errors="coerce") if error else np.nan
        native = normalized[normalized["suite"].str.contains("native", case=False, na=False)]
        query = normalized[normalized["suite"].str.contains("query|memory", case=False, na=False)]
        if not native.empty and native[["latency_ms", "error"]].notna().all(axis=1).any():
            result["native"] = native.dropna(subset=["latency_ms", "error"])
            result["native_source"] = summary_path
        if not query.empty and query["N"].notna().any():
            result["query"] = query.dropna(subset=["N"])
            result["query_source"] = summary_path
    nfe_path = _latest(root.glob("*/nfe_error.csv")) if root.exists() else None
    if nfe_path:
        raw = pd.read_csv(nfe_path)
        nfe = _column(raw, "measured_nfe", "nfe")
        error = _column(raw, "unobserved_mean_error", "error", "relative_l2")
        if nfe and error:
            result["nfe"] = pd.DataFrame(
                {"method": "DMF-Gen", "nfe": pd.to_numeric(raw[nfe], errors="coerce"), "error": pd.to_numeric(raw[error], errors="coerce")}
            ).dropna()
            result["nfe_source"] = nfe_path
    return result


def _load_draft_cost(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    pareto_path = _repo_path(repo_root, config["draft_inputs"]["architecture_cost_proxy"])
    scaling_path = _repo_path(repo_root, config["draft_inputs"]["query_scaling_proxy"])
    pareto = pd.read_csv(pareto_path)
    native = pareto[
        ["candidate", "persistent_1m_nfe4_s", "recon_nfe4_mean", "recon_nfe4_worst"]
    ].rename(columns={"candidate": "method", "persistent_1m_nfe4_s": "latency_s", "recon_nfe4_mean": "error"})
    scaling = pd.read_csv(scaling_path)
    query = scaling[scaling["execution_mode"].astype(str).eq("cached_streamed")].copy()
    query = query.rename(columns={"N_query": "N", "gpu_peak_allocated_mb": "memory_mib"})
    query["latency_ms"] = query["wall_s"] * 1000.0
    nfe_rows = []
    for row in pareto.to_dict("records"):
        for nfe in (1, 4):
            nfe_rows.append({"method": row["candidate"], "nfe": nfe, "error": row[f"recon_nfe{nfe}_mean"]})
    return {
        "native": native,
        "query": query[["N", "latency_ms", "memory_mib", "wall_s"]],
        "nfe": pd.DataFrame(nfe_rows),
        "native_source": pareto_path,
        "query_source": scaling_path,
        "nfe_source": pareto_path,
    }


def load_figure5_data(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[SourceRecord]]:
    """Load formal products when available and otherwise honest real-data draft proxies."""
    field = str(config["figure"]["field"])
    formal_uq = _load_formal_uq(config, repo_root, field)
    draft_uq = _load_draft_uq(config, repo_root, field)
    formal_cost = _load_formal_cost(config, repo_root)
    draft_cost = _load_draft_cost(config, repo_root)

    uq_map_formal = formal_uq.get("map") is not None and np.isfinite(formal_uq["map"]["std"]).any()
    coverage_formal = formal_uq.get("coverage") is not None
    spread_formal = formal_uq.get("spread_error") is not None
    native_formal = formal_cost.get("native") is not None
    query_formal = formal_cost.get("query") is not None
    nfe_formal = formal_cost.get("nfe") is not None

    data = {
        "uq_map": formal_uq["map"] if uq_map_formal else draft_uq["map"],
        "coverage": formal_uq.get("coverage"),
        "spread_error": formal_uq["spread_error"] if spread_formal else draft_uq["spread_error"],
        "cost_native": formal_cost["native"] if native_formal else draft_cost["native"],
        "cost_query": formal_cost["query"] if query_formal else draft_cost["query"],
        "cost_nfe": formal_cost["nfe"] if nfe_formal else draft_cost["nfe"],
        "modes": {
            "a": "formal" if uq_map_formal else "proxy",
            "b": "formal" if coverage_formal else "pending",
            "c": "formal" if coverage_formal and formal_uq["coverage"]["width"].notna().any() else "pending",
            "d": "formal" if spread_formal else "proxy",
            "e": "formal" if native_formal else "proxy",
            "f": "formal" if query_formal else "proxy",
            "g": "formal" if query_formal else "proxy",
            "h": "formal" if nfe_formal else "proxy",
        },
    }
    data["sources"] = {
        "a": formal_uq.get("map_source") if uq_map_formal else draft_uq["map_source"],
        "b": formal_uq.get("coverage_source", formal_uq["root"]),
        "c": formal_uq.get("coverage_source", formal_uq["root"]),
        "d": formal_uq.get("spread_error_source") if spread_formal else draft_uq["proxy_sources"][0],
        "e": formal_cost.get("native_source") if native_formal else draft_cost["native_source"],
        "f": formal_cost.get("query_source") if query_formal else draft_cost["query_source"],
        "g": formal_cost.get("query_source") if query_formal else draft_cost["query_source"],
        "h": formal_cost.get("nfe_source") if nfe_formal else draft_cost["nfe_source"],
    }
    notes = {
        "a": "Predictive ensemble map" if uq_map_formal else "Deterministic NFE=2 reconstruction; cross-NFE standard deviation is a solver-sensitivity proxy.",
        "b": "Formal S=64 state-level empirical coverage." if coverage_formal else "No formal coverage_by_level.csv found; panel intentionally remains pending.",
        "c": "Formal physical-unit interval width." if data["modes"]["c"] == "formal" else "No formal predictive interval-width table found; panel intentionally remains pending.",
        "d": "Formal state-level spread/error association." if spread_formal else "One-state spatial association between cross-NFE sensitivity and NFE=2 error; layout proxy only.",
        "e": "Formal eight-method native benchmark." if native_formal else "Architecture-level 1M-query NFE=4 Pareto; not the planned eight-method native benchmark.",
        "f": "Formal adopted-checkpoint query scaling." if query_formal else "Cached-streamed real-checkpoint systems scaling includes throughput extensions beyond the native mesh.",
        "g": "Formal adopted-checkpoint memory scaling." if query_formal else "Cached-streamed peak allocated memory proxy with throughput extensions.",
        "h": "Formal adopted-checkpoint measured-NFE sweep." if nfe_formal else "Two-point architecture-checkpoint NFE diagnostic; not the adopted DMF validation sweep.",
    }
    records = [
        SourceRecord(panel=p, mode=data["modes"][p], status="available" if data["modes"][p] != "pending" else "missing", source=str(data["sources"][p]), note=notes[p])
        for p in "abcdefgh"
    ]
    return data, records

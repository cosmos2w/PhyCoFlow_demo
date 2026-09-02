"""Strict source adapter for the metric-matched Zero-H-balanced Figure 5 backup."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) & 0x7FFFFFFF


def _load_run(directory: Path, schema: str, tables: list[str]) -> dict[str, Any]:
    required = ["manifest.json", "qa.json", *tables]
    missing = [str(directory / name) for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing formal Zero-H source(s): {missing}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    qa = json.loads((directory / "qa.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != schema or manifest.get("status") != "complete" or manifest.get("formal") is not True:
        raise ValueError(f"Run is not complete formal {schema}: {directory}")
    if qa.get("status") != "pass" or qa.get("formal") is not True:
        raise ValueError(f"Run QA is not formal/pass: {directory}")
    return {"directory": directory, "manifest": manifest, "qa": qa}


def _bootstrap_samples(states: pd.DataFrame, summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    spec = config["bootstrap"]
    namespace = str(config.get("run_namespace", "zeroh"))
    rows: list[dict[str, Any]] = []
    alpha = (1.0 - float(spec["confidence_level"])) / 2.0
    for method in config["scenario"]["generative_methods"]:
        group = states.loc[states["method"].astype(str).eq(method)].sort_values("state")
        spread = group["normalized_spread"].to_numpy(dtype=float)
        error = group["ensemble_mean_relative_l2"].to_numpy(dtype=float)
        rng = np.random.default_rng(_seed(int(spec["seed"]), f"{namespace}|rho|{method}"))
        values = np.empty(int(spec["replicates"]), dtype=float)
        for replicate in range(len(values)):
            selected = rng.integers(0, len(group), size=len(group))
            values[replicate] = float(spearmanr(spread[selected], error[selected]).statistic)
            rows.append({"method": method, "replicate": replicate, "spearman_rho": values[replicate], "sample_kind": "unique_case_pair_bootstrap"})
        expected = summary.loc[summary["method"].astype(str).eq(method)].iloc[0]
        observed = np.nanquantile(values, [alpha, 1.0 - alpha])
        if not np.allclose(observed, [float(expected["spearman_ci_low"]), float(expected["spearman_ci_high"])], rtol=0.0, atol=5e-13):
            raise ValueError(f"Bootstrap reconstruction differs for {method}")
    return pd.DataFrame(rows)


def load_zeroh_matched_v42(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    uq_dir = _resolve(repo_root, config["uq_output_root"]) / config["formal_runs"]["uq_run_id"]
    cost_dir = _resolve(repo_root, config["cost_output_root"]) / config["formal_runs"]["cost_run_id"]
    uq = _load_run(uq_dir, str(config.get("uq_schema_version", "figure5-zeroh-uq-v4.2-1")), ["per_state_method.csv", "crps_summary.csv", "spread_error_summary.csv"])
    cost = _load_run(cost_dir, str(config.get("cost_schema_version", "figure5-zeroh-cost-v4.2-1")), ["native_cost_summary.csv", "training_update_summary.csv"])
    states = pd.read_csv(uq_dir / "per_state_method.csv")
    crps = pd.read_csv(uq_dir / "crps_summary.csv")
    spread = pd.read_csv(uq_dir / "spread_error_summary.csv")
    native = pd.read_csv(cost_dir / "native_cost_summary.csv")
    training = pd.read_csv(cost_dir / "training_update_summary.csv")
    if list(crps["method"].astype(str)) != config["scenario"]["generative_methods"] or list(spread["method"].astype(str)) != config["scenario"]["generative_methods"]:
        raise ValueError("Zero-H UQ method order differs from the two adopted generative methods")
    if len(states) != int(config["cohort"]["uq_states"]) * len(config["scenario"]["generative_methods"]) or states["case_id"].nunique() != int(config["cohort"]["uq_states"]):
        raise ValueError("Scenario UQ cohort is not the configured paired unique-case cohort")
    if list(native["method"].astype(str)) != config["scenario"]["all_methods"] or list(training["method"].astype(str)) != config["scenario"]["all_methods"]:
        raise ValueError("Zero-H cost method order differs from the four adopted checkpoints")
    if set(native["status"].astype(str)) != {"ok"} or set(training["status"].astype(str)) != {"ok"}:
        raise ValueError("Matched Zero-H cost source contains an unavailable method")
    for table, metric, unit in ((native, "warm_model_core_latency_ms", "ms"), (training, "training_update_time_ms", "ms/update")):
        table["cost_metric"] = metric
        table["cost_unit"] = unit
        table["device_count"] = 1
        if not np.isfinite(table[["cost_value", "cost_low", "cost_high", "error", "error_ci_low", "error_ci_high"]].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite Zero-H {metric} source")
    crps_samples = states[["method", "state", "case_id", "time_index", "normalized_crps"]].copy()
    crps_samples["sample_kind"] = "paired_unique_case_time_state"
    rho_samples = _bootstrap_samples(states, spread, config)
    return {
        "uq_crps_samples": crps_samples,
        "uq_spearman_bootstrap": rho_samples,
        "uq_crps": crps,
        "uq_spread": spread,
        "cost_native": native,
        "training_cost": training,
        "training_metric_label": f"Canonical {'/'.join(config['scenario'].get('training_resolutions', ['L', 'M']))} training update time (ms/update)",
        "uq": uq,
        "cost": cost,
    }


load_superres_matched = load_zeroh_matched_v42

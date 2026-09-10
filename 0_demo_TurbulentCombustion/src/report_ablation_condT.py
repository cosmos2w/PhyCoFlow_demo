#!/usr/bin/env python
"""Render the reviewed Cond_T ablation evaluation tables and protocol.

The evaluator writes one directory per checkpoint policy (``last`` and
``best``). This renderer keeps those policies separate while also pairing
their per-state records. The default report uses last.pt as the primary
checkpoint and includes a complete best.pt table in every metric section.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEMO = Path(__file__).resolve().parents[1]
ROOT = DEMO / "Save_TrainedModel/ablation_condT"
FIELDS = ["CH4", "CO", "T", "U1", "p"]
MODELS = [f"A{i}" for i in range(6)]
POLICIES = ("last", "best")
DATE_RE = re.compile(r"(?:evaluation[_-])?(\d{8})$")
LABELS = {
    "A0": "Newly trained GL-RBF-ENH baseline (A0)",
    "A1": "Deterministic direct-field regression, same backbone",
    "A2": "No global-latent feedback into sensor tokens",
    "A3": "No local query-to-sensor Top-K/RBF conditioning",
    "A4": "IID Gaussian prior instead of spatial RFF prior",
    "A5": "Local sensor-token Top-K/RBF observation route only",
}


def table(headers, rows):
    """Return a small GitHub-flavoured Markdown table."""
    return "\n".join(
        [
            "| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def read_json(path: Path, default=None):
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def infer_evaluation_root(path: Path) -> Path:
    """Accept either the policy root or a policy/metrics directory."""
    path = path.resolve()
    if path.name == "metrics" and path.parent.name in POLICIES:
        return path.parent.parent
    if path.name in POLICIES and (path / "metrics").is_dir():
        return path.parent
    return path


def default_evaluation_root() -> Path:
    candidates = sorted(
        (p for p in ROOT.glob("evaluation_*") if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for candidate in candidates:
        if all((candidate / policy / "metrics").is_dir() for policy in POLICIES):
            return candidate
    for candidate in candidates:
        if (candidate / "last" / "metrics").is_dir():
            return candidate
    return ROOT / f"evaluation_{date.today():%Y%m%d}"


def find_metrics_dir(root: Path, policy: str, explicit: Path | None = None) -> Path:
    """Locate a policy's metrics directory across old and new layouts."""
    candidates = []
    if explicit is not None:
        explicit = explicit.resolve()
        candidates.extend(
            [
                explicit / "metrics",
                explicit,
                explicit / policy / "metrics",
                explicit / policy,
            ]
        )
    candidates.extend(
        [
            root / policy / "metrics",
            root / policy,
            root / "metrics" if root.name == policy else root / "__missing__",
        ]
    )
    for candidate in candidates:
        if (candidate / "summary_metrics.csv").is_file() and (
            candidate / "per_state_metrics.csv"
        ).is_file():
            return candidate.resolve()
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Missing {policy} metrics directory; searched {searched}")


def _numeric(frame: pd.DataFrame, columns):
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise")


def load_policy(root: Path, policy: str, explicit: Path | None, expected: int, allow_partial: bool):
    metrics_dir = find_metrics_dir(root, policy, explicit)
    summary = pd.read_csv(metrics_dir / "summary_metrics.csv")
    per = pd.read_csv(metrics_dir / "per_state_metrics.csv")
    paired_path = metrics_dir / "paired_differences.csv"
    paired = pd.read_csv(paired_path) if paired_path.is_file() else pd.DataFrame()
    meta = read_json(metrics_dir / "analysis_metadata.json")
    _numeric(summary, ["n", "mean", "std", "median", "q25", "q75", "p95", "min", "max"])
    _numeric(summary, ["iid_ci95_low", "iid_ci95_high", "block20_ci95_low", "block20_ci95_high"])
    _numeric(per, ["snapshot", "time_index", "value"])
    _numeric(
        paired,
        [
            "n_paired",
            "method_mean",
            "baseline_mean",
            "mean_difference",
            "relative_mean_change_percent",
            "fraction_method_less_than_baseline",
            "fraction_method_equal_baseline",
            "block5_ci95_low",
            "block5_ci95_high",
            "block20_ci95_low",
            "block20_ci95_high",
            "block50_ci95_low",
            "block50_ci95_high",
        ],
    )
    result = {
        "name": policy,
        "dir": metrics_dir,
        "summary": summary,
        "per": per,
        "paired": paired,
        "meta": meta,
        "full": False,
        "expected": expected,
    }
    validate_policy(result, expected, allow_partial)
    return result


def validate_policy(policy, expected: int, allow_partial: bool):
    """Reject silently incomplete or non-paired metric data."""
    summary = policy["summary"]
    per = policy["per"]
    meta = policy["meta"]
    expected_methods = set(MODELS)
    methods = set(summary["method"].astype(str))
    if methods != expected_methods:
        raise ValueError(f"{policy['name']}: summary methods are {sorted(methods)}, expected {MODELS}")
    if not {"method", "metric", "target", "mean"}.issubset(summary.columns):
        raise ValueError(f"{policy['name']}: summary_metrics.csv lacks required columns")
    if not {"method", "snapshot", "time_index", "metric", "target", "value"}.issubset(per.columns):
        raise ValueError(f"{policy['name']}: per_state_metrics.csv lacks required columns")
    keys = ["method", "snapshot", "metric", "target"]
    if per.duplicated(keys).any():
        raise ValueError(f"{policy['name']}: duplicate per-state metric identities")
    counts = per.groupby("method")["snapshot"].nunique().to_dict()
    metadata_counts = meta.get("snapshots_by_method", {})
    expected_metadata = meta.get("expected_snapshots_per_method")
    declared_full = (
        not bool(meta.get("partial", False))
        and (meta.get("cache_count") in (None, 6 * expected))
        and all(int(counts.get(method, -1)) == expected for method in MODELS)
        and all(int(metadata_counts.get(method, expected)) == expected for method in MODELS)
        and (expected_metadata in (None, expected))
    )
    policy["counts"] = {method: int(counts.get(method, 0)) for method in MODELS}
    policy["full"] = declared_full
    if not allow_partial and not declared_full:
        raise ValueError(
            f"{policy['name']}: incomplete evaluation {policy['counts']} or partial metadata; "
            f"a final report requires all six methods x {expected} states"
        )


def report_date(root: Path, requested: str | None) -> str:
    if requested:
        digits = re.sub(r"[^0-9]", "", requested)
        if len(digits) == 8:
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
        return requested
    match = DATE_RE.search(root.name)
    if match:
        digits = match.group(1)
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return f"{date.today():%Y-%m-%d}"


def value(policy, method: str, metric: str, target: str):
    frame = policy["summary"]
    rows = frame[
        (frame.method.astype(str) == method)
        & (frame.metric.astype(str) == metric)
        & (frame.target.astype(str) == target)
    ]
    if len(rows) != 1:
        raise KeyError(f"{policy['name']}: expected one summary row for {method}/{metric}/{target}, got {len(rows)}")
    return rows.iloc[0]


def mean(policy, method: str, metric: str, target: str) -> float:
    return float(value(policy, method, metric, target)["mean"])


def format_value(raw, decimals=5, scale=1.0):
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return "n/a"
    try:
        number = float(raw) * scale
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{decimals}f}"


def format_general(raw, significant=3):
    """Format small numerical audit values without rounding them to zero."""
    if raw is None:
        return "n/a"
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number:.{significant}g}"


def ci(row, prefix="block20", decimals=5):
    try:
        low, high = float(row[f"{prefix}_ci95_low"]), float(row[f"{prefix}_ci95_high"])
    except (KeyError, TypeError, ValueError):
        return "n/a"
    if not np.isfinite(low) or not np.isfinite(high):
        return "n/a"
    return f"[{low:.{decimals}f}, {high:.{decimals}f}]"


def matrix(policy, metric, targets, scale=1.0, decimals=5):
    return table(
        ["Run"] + targets,
        [
            [method]
            + [format_value(mean(policy, method, metric, target), decimals, scale) for target in targets]
            for method in MODELS
        ],
    )


def dual_matrix(policies, metric, targets, scale=1.0, decimals=5):
    """Keep explicit last/best tables so the checkpoint policy is visible."""
    return (
        "**last.pt:**\n\n"
        + matrix(policies["last"], metric, targets, scale, decimals)
        + "\n\n**best.pt:**\n\n"
        + matrix(policies["best"], metric, targets, scale, decimals)
    )


def combined_matrix(policies, metric, targets, scale=1.0, decimals=5):
    headers = ["Run"] + [f"{target} last" for target in targets] + [f"{target} best" for target in targets]
    rows = []
    for method in MODELS:
        rows.append(
            [method]
            + [format_value(mean(policies["last"], method, metric, target), decimals, scale) for target in targets]
            + [format_value(mean(policies["best"], method, metric, target), decimals, scale) for target in targets]
        )
    return table(headers, rows)


def extrema_matrix(policy, metric, targets, statistic="min", decimals=6):
    """Render a true summary extremum, rather than the mean of state extrema."""
    return table(
        ["Run"] + targets,
        [
            [method]
            + [format_value(value(policy, method, metric, target).get(statistic), decimals) for target in targets]
            for method in MODELS
        ],
    )


def dual_extrema_matrix(policies, metric, targets, statistic="min", decimals=6):
    return (
        "**last.pt:**\n\n"
        + extrema_matrix(policies["last"], metric, targets, statistic, decimals)
        + "\n\n**best.pt:**\n\n"
        + extrema_matrix(policies["best"], metric, targets, statistic, decimals)
    )


def nonphysical_count_table(policies, n_points):
    """Convert per-state mean fractions to counts over the evaluated points."""
    rows = []
    for policy in POLICIES:
        for method in MODELS:
            if not isinstance(n_points, int):
                t_count = p_count = "n/a"
            else:
                total = n_points * policies[policy]["counts"].get(method, 0)
                t_count = round(mean(policies[policy], method, "reconstruction_nonphysical_fraction", "T") * total)
                p_count = round(mean(policies[policy], method, "reconstruction_nonphysical_fraction", "p") * total)
            rows.append([policy, method, t_count, p_count])
    return table(["Policy", "Run", "T ≤ 0 count", "p ≤ 0 count"], rows)


def diffs(policy, metric, targets, baseline="A0"):
    paired = policy["paired"]
    if paired.empty:
        return "No paired-difference file was provided."
    selected = paired[
        (paired.metric.astype(str) == metric)
        & (paired.target.astype(str).isin(targets))
        & (paired.baseline.astype(str) == baseline)
    ]
    rows = []
    for _, item in selected.iterrows():
        rel = item.get("relative_mean_change_percent")
        fraction = item.get("fraction_method_less_than_baseline")
        rel_text = "n/a" if pd.isna(rel) else f"{float(rel):+.1f}%"
        fraction_text = "n/a" if pd.isna(fraction) else f"{100 * float(fraction):.1f}%"
        rows.append(
            [
                f"{item.method} − {item.baseline}",
                item.target,
                f"{float(item.mean_difference):+.5f}",
                ci(item),
                rel_text,
                fraction_text,
            ]
        )
    return table(
        [
            "Run − reference",
            "Target",
            "Mean difference",
            "95% block-20 CI",
            "Relative mean change",
            "Fraction lower",
        ],
        rows,
    )


def dual_diffs(policies, metric, targets, baseline="A0"):
    return (
        "**last.pt:**\n\n"
        + diffs(policies["last"], metric, targets, baseline)
        + "\n\n**best.pt:**\n\n"
        + diffs(policies["best"], metric, targets, baseline)
    )


def block_bootstrap_means(values, *, block: int, n_boot: int, seed: int):
    """Circular moving-block bootstrap of temporally sorted state values."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite nonempty values")
    block = min(int(block), len(values))
    n_blocks = (len(values) + block - 1) // block
    starts = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, n_blocks))
    indices = ((starts[:, :, None] + np.arange(block)) % len(values)).reshape(n_boot, -1)[:, : len(values)]
    return values[indices].mean(axis=1)


def checkpoint_sensitivity(policies, root: Path, n_boot: int, seed: int, allow_partial: bool = False):
    """Pair every common best/last state and compute uncertainty on deltas."""
    keys = ["method", "snapshot", "metric", "target"]
    identity = keys + ["time_index"]
    last = policies["last"]["per"][identity + ["value"]].rename(columns={"value": "last_value"})
    best = policies["best"]["per"][identity + ["value"]].rename(columns={"value": "best_value"})
    matched = best.merge(last, on=identity, how="inner", validate="one_to_one")
    if len(matched) != len(best) or len(matched) != len(last):
        if allow_partial:
            # Compatibility mode is useful for inspecting the historical
            # evaluation, whose best policy covered a fixed subset. A final
            # report never takes this path because both policies are required
            # to be full.
            pass
        else:
            missing_best = len(best) - len(matched)
            missing_last = len(last) - len(matched)
            raise ValueError(
                f"Best/last per-state coverage is not identical ({missing_best} best-only, {missing_last} last-only)"
            )
    rows = []
    for (method, metric, target), group in matched.groupby(["method", "metric", "target"], sort=True):
        group = group.sort_values(["time_index", "snapshot"])
        delta = group.best_value.to_numpy(dtype=float) - group.last_value.to_numpy(dtype=float)
        low_high = {}
        for block in (5, 20, 50):
            boot = block_bootstrap_means(delta, block=block, n_boot=n_boot, seed=seed)
            low_high[f"block{block}_ci95_low"], low_high[f"block{block}_ci95_high"] = map(
                float, np.quantile(boot, [0.025, 0.975])
            )
            low_high[f"block{block}_excludes_zero"] = bool(
                low_high[f"block{block}_ci95_low"] > 0 or low_high[f"block{block}_ci95_high"] < 0
            )
        last_mean = float(group.last_value.mean())
        best_mean = float(group.best_value.mean())
        rows.append(
            {
                "method": method,
                "metric": metric,
                "target": target,
                "n_paired": int(len(group)),
                "last_mean": last_mean,
                "best_mean": best_mean,
                "best_minus_last": float(delta.mean()),
                "relative_mean_change_percent": float(100 * delta.mean() / last_mean) if last_mean else np.nan,
                "fraction_best_less_than_last": float(np.mean(delta < 0)),
                "fraction_best_equal_last": float(np.mean(delta == 0)),
                "snapshot_min": int(group.snapshot.min()),
                "snapshot_max": int(group.snapshot.max()),
                **low_high,
            }
        )
    output = pd.DataFrame(rows)
    path = root / "checkpoint_sensitivity_vs_last.csv"
    output.to_csv(path, index=False)
    return output, path


def sensitivity_rows(sensitivity, metric, targets, decimals=5):
    selected = sensitivity[
        (sensitivity.metric.astype(str) == metric) & (sensitivity.target.astype(str).isin(targets))
    ]
    rows = []
    for _, item in selected.iterrows():
        rows.append(
            [
                item.method,
                item.target,
                format_value(item.last_mean, decimals),
                format_value(item.best_mean, decimals),
                f"{float(item.best_minus_last):+.{decimals}f}",
                f"[{float(item.block20_ci95_low):+.{decimals}f}, {float(item.block20_ci95_high):+.{decimals}f}]",
                f"{100 * float(item.fraction_best_less_than_last):.1f}%",
            ]
        )
    return table(
        ["Run", "Target", "Last mean", "Best mean", "Best − last", "95% block-20 CI", "Best lower"],
        rows,
    )


def load_audit(root: Path):
    return read_json(root / "checkpoint_audit.json", {"runs": []})


def load_provenance(policy, method: str):
    path = policy["dir"].parent / f"provenance_{method}.json"
    return read_json(path, {})


def audit_run_map(audit):
    return {str(item.get("id")): item for item in audit.get("runs", []) if item.get("id")}


def checkpoint_info(audit_map, policies, method: str, policy: str):
    run = audit_map.get(method, {})
    provenance = load_provenance(policies[policy], method)
    source = run if run else provenance
    checkpoints = source.get("checkpoints", {})
    item = checkpoints.get(f"{policy}.pt", {}) or checkpoints.get(policy, {}) or {}
    if not item:
        item = checkpoints.get("last.pt", {}) if policy == "last" else checkpoints.get("best.pt", {})
    return item, source, provenance


def first_provenance(audit_map, policies):
    for method in MODELS:
        for policy in POLICIES:
            _, source, provenance = checkpoint_info(audit_map, policies, method, policy)
            # The root audit contains checkpoint metadata, while the policy
            # provenance contains the actual inference protocol (device,
            # grid, sensor plan, and requested snapshots). Prefer the latter.
            if provenance:
                return provenance
            if source:
                return source
    return {}


def infer_device(root: Path, policies, requested: str | None):
    if requested:
        return requested
    environment = read_json(root / "evaluation_environment.json", {})
    for key in ("device", "inference_device", "gpu", "cuda_device"):
        if key in environment and environment[key] not in (None, ""):
            value = environment[key]
            if isinstance(value, int):
                return f"cuda:{value}"
            return str(value)
    for policy in POLICIES:
        provenance = load_provenance(policies[policy], "A0")
        if provenance.get("device"):
            return str(provenance["device"])
    return "cuda:0"


def protocol_context(policies, audit_map):
    provenance = first_provenance(audit_map, policies)
    fields = provenance.get("field_names", FIELDS)
    fields = [str(field).replace("_", "") for field in fields]
    num_x = provenance.get("num_x", 403)
    num_y = provenance.get("num_y", 100)
    try:
        n_points = int(num_x) * int(num_y)
    except (TypeError, ValueError):
        n_points = "all grid points"
    requested = provenance.get("requested_snapshots") or provenance.get("test_time_indices") or []
    n_states = len(requested) if requested else policies["last"]["counts"].get("A0", 1000)
    sensor_plan = provenance.get("sensor_plan", "the archived SensorPlan_paper_full_20260711.csv")
    n_obs = 256
    if sensor_plan and Path(str(sensor_plan)).is_file():
        try:
            with Path(str(sensor_plan)).open(newline="", encoding="utf-8") as stream:
                rows = [row for row in csv.DictReader(stream) if row.get("condition") == "Cond_T"]
            if rows:
                first_snapshot = rows[0].get("snapshot")
                n_obs = len([row for row in rows if row.get("snapshot") == first_snapshot])
        except (OSError, csv.Error):
            pass
    return {
        "provenance": provenance,
        "fields": fields,
        "num_x": num_x,
        "num_y": num_y,
        "n_points": n_points,
        "n_states": n_states,
        "n_obs": n_obs,
        "sensor_plan": sensor_plan,
    }


def section(chunks, title, text):
    chunks.append(f"## {title}\n\n{text}")


def variability_table(policy, metric="physical_relative_l2", target="Unobserved_mean"):
    return table(
        ["Run", "Median", "25th–75th percentile", "95th percentile", "Maximum"],
        [
            [
                method,
                format_value(value(policy, method, metric, target)["median"]),
                f"{format_value(value(policy, method, metric, target)['q25'])}–{format_value(value(policy, method, metric, target)['q75'])}",
                format_value(value(policy, method, metric, target)["p95"]),
                format_value(value(policy, method, metric, target)["max"]),
            ]
            for method in MODELS
        ],
    )


def dual_variability(policies, metric="physical_relative_l2", target="Unobserved_mean"):
    return (
        "**last.pt:**\n\n"
        + variability_table(policies["last"], metric, target)
        + "\n\n**best.pt:**\n\n"
        + variability_table(policies["best"], metric, target)
    )


def sensor_table(policy):
    return table(
        ["Run", "T physical L2, all points", "T physical L2, unobserved points only", "Maximum sensor absolute error"],
        [
            [
                method,
                format_value(mean(policy, method, "physical_relative_l2", "T"), 6),
                format_value(mean(policy, method, "physical_relative_l2_excluding_sensors", "T"), 6),
                format_value(value(policy, method, "sensor_max_abs_normalized_error", "T")["max"], 3),
            ]
            for method in MODELS
        ],
    )


def dual_sensor_table(policies):
    return "**last.pt:**\n\n" + sensor_table(policies["last"]) + "\n\n**best.pt:**\n\n" + sensor_table(policies["best"])


def provenance_rows(audit_map, policies):
    rows = []
    for method in MODELS:
        last_item, last_source, last_prov = checkpoint_info(audit_map, policies, method, "last")
        best_item, _, _ = checkpoint_info(audit_map, policies, method, "best")
        parameter_elements = (
            last_item.get("parameter_elements_with_optimizer_state")
            or last_source.get("parameter_elements_with_optimizer_state")
            or last_prov.get("parameter_elements_with_optimizer_state")
        )
        if parameter_elements is None:
            parameter_elements = "n/a"
        elif isinstance(parameter_elements, (int, float)):
            parameter_elements = f"{int(parameter_elements):,}"
        rows.append(
            [
                method,
                LABELS[method],
                last_item.get("epoch", "n/a"),
                best_item.get("epoch", "n/a"),
                parameter_elements,
            ]
        )
    return rows


def equivalence_rows(audit_map, policies):
    rows = []
    for policy in POLICIES:
        for method in MODELS:
            item, source, provenance = checkpoint_info(audit_map, policies, method, policy)
            equivalence = provenance.get("execution_equivalence", source.get("execution_equivalence", {}))
            rows.append(
                [
                    method,
                    policy,
                    equivalence.get("passed", "n/a"),
                    format_general(equivalence.get("max_abs_normalized")),
                    format_general(equivalence.get("relative_l2")),
                    provenance.get("completed_snapshots", "n/a"),
                    item.get("epoch", "n/a"),
                ]
            )
    return rows


def historical_crosscheck_table(policies):
    rows = []
    for policy in POLICIES:
        path = policies[policy]["dir"] / "A0_paper_crosscheck.csv"
        if not path.is_file():
            continue
        checks = pd.read_csv(path)
        if checks.empty:
            continue
        for metric, group in checks.groupby("metric"):
            rows.append(
                [
                    policy,
                    metric,
                    len(group),
                    f"{group.absolute_difference.mean():.3g}",
                    f"{group.absolute_difference.max():.3g}",
                ]
            )
    return table(
        ["Policy", "Metric", "Compared values", "Mean absolute difference", "Maximum absolute difference"],
        rows,
    )


def nonpositive_events(policy):
    per = policy["per"]
    selected = per[
        (per.metric.astype(str) == "reconstruction_minimum")
        & (per.target.astype(str) == "T")
        & (per.value <= 0)
    ]
    return selected.sort_values(["method", "snapshot"])


def worst_states(policy):
    per = policy["per"]
    return (
        per[(per.metric.astype(str) == "physical_relative_l2") & (per.target.astype(str) == "Unobserved_mean")]
        .sort_values("value", ascending=False)
        .groupby("method", sort=True)
        .head(3)
        .sort_values(["method", "value"], ascending=[True, False])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        help="Root containing last/ and best/ policy directories; defaults to the newest evaluation_* root",
    )
    parser.add_argument("--last-evaluation-dir", type=Path, help="Optional explicit last-policy directory")
    parser.add_argument("--best-evaluation-dir", type=Path, help="Optional explicit best-policy directory")
    parser.add_argument("--evaluation-date", "--date", dest="evaluation_date", help="Report date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--expected-snapshots", type=int, default=1000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, help="Bootstrap seed; defaults to the seed recorded in analysis metadata")
    parser.add_argument("--device", help="Device text used in reproduction commands; otherwise read evaluation metadata")
    parser.add_argument("--allow-partial", action="store_true", help="Allow explicitly incomplete policy data for smoke reports")
    parser.add_argument("--output", type=Path, help="Markdown output path")
    args = parser.parse_args()

    root = infer_evaluation_root(args.evaluation_dir or args.last_evaluation_dir or default_evaluation_root())
    root.mkdir(parents=True, exist_ok=True)
    policies = {
        "last": load_policy(root, "last", args.last_evaluation_dir, args.expected_snapshots, args.allow_partial),
        "best": load_policy(root, "best", args.best_evaluation_dir, args.expected_snapshots, args.allow_partial),
    }
    if not args.allow_partial and not all(policies[policy]["full"] for policy in POLICIES):
        raise ValueError("A final report requires complete last.pt and best.pt evaluations for all six models")
    evaluation_date = report_date(root, args.evaluation_date)
    output = (args.output or root / f"Ablation_CondT_Evaluation_{evaluation_date.replace('-', '')}.md").resolve()
    audit = load_audit(root)
    audit_map = audit_run_map(audit)
    context = protocol_context(policies, audit_map)
    device = infer_device(root, policies, args.device)
    primary = policies["last"]
    metadata_seed = primary["meta"].get("bootstrap", {}).get("seed")
    if args.seed is not None:
        bootstrap_seed = args.seed
    elif metadata_seed is not None:
        bootstrap_seed = int(metadata_seed)
    else:
        bootstrap_seed = int(evaluation_date.replace("-", ""))
    population_label = (
        f"all {args.expected_snapshots:,} held-out snapshots"
        if all(policies[policy]["full"] for policy in POLICIES)
        else "the available paired held-out snapshots (partial smoke-report mode)"
    )

    sensitivity, sensitivity_path = checkpoint_sensitivity(
        policies, root, args.bootstrap_resamples, bootstrap_seed, args.allow_partial
    )

    chunks = []
    chunks.append(
        f"# Cond_T ablation evaluation: reconstruction and physical fidelity\n\n"
        f"Evaluation date: {evaluation_date}. Complete inference-only evaluation of six saved checkpoint variants over {population_label}. "
        f"No training, checkpoint modification, or clipping of predicted fields was performed. Inference device: `{device}`; metrics were computed on CPU.\n\n"
        "Primary checkpoint policy: **last.pt**. The complete **best.pt** evaluation is reported alongside it and is paired state by state for checkpoint sensitivity. "
        "The A0 checkpoint in this evaluation is the newly trained A0 run; the archived paper output is retained only as a historical comparison."
    )
    reference = DEMO / "Save_TrainedModel/_TrainedModels/_Process_Figures/Assembled/Composite/CoupledFieldReconstruction_round3_spacing_20260828_0810.pdf"
    prior_reports = sorted(root.glob("Ablation_CondT_Evaluation_*.previous.md"))
    archived_report = prior_reports[-1] if prior_reports else root / "Ablation_CondT_Evaluation_20260906.original.md"
    archived_report_note = (
        f" The superseded report is preserved at [{archived_report.name}]({archived_report.resolve()})."
        if archived_report.is_file()
        else " The prior report should be preserved separately before publishing this replacement."
    )
    a0_input = root / "checkpoint_inputs" / "A0"
    a0_input_note = (
        f" The frozen A0 checkpoint inputs are recorded at [checkpoint_inputs/A0]({a0_input.resolve()})."
        if a0_input.is_dir()
        else ""
    )
    chunks.append(
        f"Reference: [CoupledFieldReconstruction round3 spacing, panels c and d]({reference}). "
        "The archived result is a definition/provenance reference; it is not treated as the A0 checkpoint for this six-way comparison. "
        "This report replaces the archived A0 baseline with the saved A0 checkpoints listed below."
        + archived_report_note
        + a0_input_note
    )

    ranking_last = sorted(MODELS, key=lambda method: mean(primary, method, "physical_relative_l2", "Unobserved_mean"))
    ranking_best = sorted(MODELS, key=lambda method: mean(policies["best"], method, "physical_relative_l2", "Unobserved_mean"))
    ranking_rows = []
    for method in MODELS:
        last_mean = mean(primary, method, "physical_relative_l2", "Unobserved_mean")
        best_mean = mean(policies["best"], method, "physical_relative_l2", "Unobserved_mean")
        ranking_rows.append(
            [
                method,
                f"{last_mean:.6f}",
                ci(value(primary, method, "physical_relative_l2", "Unobserved_mean"), decimals=6),
                f"{100 * (last_mean / mean(primary, 'A0', 'physical_relative_l2', 'Unobserved_mean') - 1):+.1f}%",
                f"{best_mean:.6f}",
                ci(value(policies["best"], method, "physical_relative_l2", "Unobserved_mean"), decimals=6),
                f"{100 * (best_mean / mean(policies['best'], 'A0', 'physical_relative_l2', 'Unobserved_mean') - 1):+.1f}%",
            ]
        )
    section(
        chunks,
        "1. Main quantitative findings",
        "Last-checkpoint ranking by mean physical relative-L2 over the four unobserved fields (lower is better): **"
        + " < ".join(ranking_last)
        + "**. Best-checkpoint ranking on the same held-out population: **"
        + " < ".join(ranking_best)
        + "**.\n\n"
        + table(
            [
                "Run",
                "Last unobserved mean",
                "Last 95% block-20 CI",
                "Last change vs A0",
                "Best unobserved mean",
                "Best 95% block-20 CI",
                "Best change vs A0",
            ],
            ranking_rows,
        )
        + "\n\nThese are measurements of saved checkpoints, not a presumption that every ablation must be worse. Spatial relative-L2, spectral error, and joint-distribution error measure different properties and are interpreted separately below.",
    )

    interpretation = root / "reviewed_interpretation.md"
    if interpretation.is_file():
        chunks.append(interpretation.read_text(encoding="utf-8").strip())

    provenance = provenance_rows(audit_map, policies)
    run_paths = []
    for method in MODELS:
        _, source, prov = checkpoint_info(audit_map, policies, method, "last")
        run_paths.append(prov.get("run_directory") or source.get("path") or "")
    distinct_param_counts = set()
    for row in provenance:
        if row[-1] != "n/a":
            try:
                distinct_param_counts.add(int(str(row[-1]).replace(",", "")))
            except ValueError:
                pass
    parameter_note = (
        f"Observed optimizer-state parameter-element counts are {', '.join(f'{n:,}' for n in sorted(distinct_param_counts))}."
        if distinct_param_counts
        else "Optimizer-state parameter-element counts were not recorded in the available audit."
    )
    section(
        chunks,
        "2. Models and checkpoint provenance",
        table(
            ["Run", "Meaning", "Last epoch", "Best epoch", "Parameter elements with optimizer state (last)"],
            provenance,
        )
        + "\n\n"
        + parameter_note
        + " Bypassed parameters may remain in module/state dictionaries without participating in optimization; optimizer-state counts are not a FLOP count.\n\n"
        "- A1 predicts the normalized field directly from zero state at fixed time, without sampling a prior or integrating a flow. Its direct-MSE training loss is not comparable with flow-velocity losses.\n"
        "- A2 removes only latent-to-sensor refinement. Sensor-to-latent encoding, latent processing, global query readout, and local query conditioning remain.\n"
        "- A3 zeros local query aggregation in both legacy and cached paths, but keeps global latent conditioning. This is a global-bottleneck decoder, not an exact implementation of a separately trained Perceiver comparator.\n"
        "- A4 changes the prior. If its random-number construction differs from the RFF path, equal training seeds do not guarantee equal initial backbone tensors. This is a saved-checkpoint comparison, not a perfectly initialization-paired causal experiment.\n"
        "- A5 additionally bypasses sensor-to-latent encoding, global summary, and query-latent readout. Observation information reaches queries only through sensor-token Top-K/RBF; it is not bare kernel interpolation.\n\n"
        "Normalization statistics, strict-load checks, architecture metadata, and checkpoint hashes are recorded in the audit/provenance files when available. The table uses epoch values stored in each checkpoint. The saved checkpoint paths recorded for this evaluation include: "
        + ", ".join(path for path in run_paths if path)
        + ".",
    )

    provenance0 = context["provenance"]
    seed_policy = provenance0.get(
        "generation_seed_policy",
        'stable_seed(20260711, "generation", "DMF-Gen", "Cond_T", snapshot)',
    )
    n_states = context["n_states"]
    n_points = context["n_points"]
    n_obs = context["n_obs"]
    protocol_text = (
        f"- Population: {n_states:,} held-out snapshots, each with {n_points:,} spatial points where grid dimensions are available, fields {', '.join(context['fields'])}. "
        "The complete report requires six methods evaluated on the same snapshot identities.\n"
        f"- Conditioning: `{context['sensor_plan']}`; {n_obs} temperature observations per state under the Cond_T plan. No velocity, pressure, or species observations. All models share the exact observations and truth arrays.\n"
        "- Generative inference: two Euler steps, one sampled reconstruction per state, `default_hard` observed-entry clamping, live checkpoint weights, no EMA substitution. A1 uses one deterministic forward pass and the same final sensor clamp.\n"
        f"- Sampling seed policy: `{seed_policy}`. Matched seeds mean matched random seeding, not identical prior fields between architectures.\n"
        "- Execution: cached/streamed inference for compatible models and the recorded legacy path for A1 where applicable. Complete-state legacy/cached equivalence is audited per policy when requested. Strict checkpoint loading and finite predictions are mandatory.\n"
        "- Aggregation: compute each field metric per snapshot, then give each snapshot equal weight. No pooling of unlike physical units and no treatment of spatial pixels as independent statistical replicates.\n\n"
        "**Important split caveat:** `TurbulentCombustionH5Dataset` maps `val` and `test` to the same shuffled held-out time indices in this project protocol. Thus this reproduces the project/paper test protocol, but it is not an untouched dataset independent of training-time validation/checkpoint selection. The snapshots also come from one temporally related simulation, not independent experiments.\n\n"
        f"Confidence intervals use {args.bootstrap_resamples:,} bootstrap resamples. Summary intervals use circular moving blocks of 20 temporally sorted held-out states; paired ablation differences and checkpoint deltas retain block lengths 5, 20, and 50 in machine-readable output. These are descriptive intervals conditional on the checkpoint, sensor plan, and single generative draw; they do not measure training-seed variability and are not multiplicity-adjusted."
    )
    if args.allow_partial:
        protocol_text += "\n\n**Partial-report flag:** at least one policy is incomplete; tables and checkpoint deltas are labeled by their available paired coverage and are not a final six-by-1,000 result."
    section(chunks, "3. Matched evaluation protocol", protocol_text)

    section(
        chunks,
        "4. Reconstruction error: physical relative-L2",
        "For each state and field, `L2 = ||prediction − truth||₂ / (||truth||₂ + 1e−12)` after restoring physical units. `Unobserved_mean` is the arithmetic mean of CH4, CO, U1, and p relative errors; `All_fields_mean` additionally includes T. Values below are fractions (0.10 = 10%), not percentages.\n\n"
        + dual_matrix(policies, "physical_relative_l2", FIELDS + ["Unobserved_mean", "All_fields_mean"])
        + "\n\n### Statewise variability of unobserved-field mean\n\n"
        + dual_variability(policies)
        + "\n\n### Paired change versus baseline\n\n"
        + dual_diffs(policies, "physical_relative_l2", FIELDS + ["Unobserved_mean"])
        + "\n\nNegative difference/change means lower error. A confidence interval crossing zero indicates no clearly resolved direction under this descriptive temporal-block analysis.",
    )

    section(
        chunks,
        "5. Normalized and sensor-excluded checks",
        "Normalized relative-L2 uses the training-standardized field, including subtraction of the training mean. It is not the same as physical relative-L2: pressure has a large mean, so physical relative-L2 can look small despite appreciable fluctuation error.\n\n"
        + dual_matrix(policies, "normalized_relative_l2", FIELDS + ["Unobserved_mean"])
        + "\n\nA complementary error divides physical error by each truth snapshot’s centered fluctuation norm, `||prediction − truth||₂ / ||truth − mean_spatial(truth)||₂`:\n\n"
        + dual_matrix(policies, "truth_fluctuation_normalized_l2", FIELDS)
        + "\n\nT error with the clamped locations removed, and maximum normalized sensor mismatch:\n\n"
        + dual_sensor_table(policies)
        + "\n\nExact sensor agreement is imposed by inference and must not be interpreted as a learned reconstruction achievement.",
    )

    section(
        chunks,
        "6. Panel c: spatial spectral fidelity",
        "The exact paper metric is the root mean square of `10 log10[(E_pred(k)+epsilon)/(E_truth(k)+epsilon)]` over retained radial shells, in dB; lower is better. Each field is spatially demeaned; no window is applied. The archived nonuniform physical mesh resolves to **topological/index-space** spectral coordinates under the canonical auto policy. Isotropic cutoff is enabled, minimum shell count is 4, and epsilon is 1e−12 times the largest truth shell energy (floor 1e−30). These measure spectral structure on mesh topology, not a nonuniform-grid physical-wavenumber spectrum or a verified inertial-range scaling law.\n\nThe reference panel displays CH4, p, and U1; all five fields are included here.\n\n"
        + dual_matrix(policies, "spectral_lsd_db", FIELDS, decimals=4)
        + "\n\nPaired differences for the three displayed fields:\n\n"
        + dual_diffs(policies, "spectral_lsd_db", ["CH4", "p", "U1"])
        + "\n\n### Spectral energy diagnostics\n\nMean reconstructed/truth total spectral energy ratio (ideal 1):\n\n"
        + dual_matrix(policies, "spectral_total_energy_ratio", FIELDS, decimals=4)
        + "\n\nMean reconstructed/truth energy in the middle third of retained wavenumbers (ideal 1):\n\n"
        + dual_matrix(policies, "spectral_mid_energy_ratio", FIELDS, decimals=4)
        + "\n\nMean reconstructed/truth energy in the highest third (ideal 1):\n\n"
        + dual_matrix(policies, "spectral_high_energy_ratio", FIELDS, decimals=4)
        + "\n\nBand energies use canonical trapezoidal integration with low/middle/high boundaries at one and two thirds of the retained maximum wavenumber. An accurate total energy can coexist with distorted small scales; matching a spectrum does not establish correct spatial phase or feature location.",
    )

    spectral_supplement = root / "high_frequency_interpretation.md"
    if spectral_supplement.is_file():
        chunks.append(spectral_supplement.read_text(encoding="utf-8").strip())

    pairs = ["T-U1", "CH4-U1", "p-U1"]
    section(
        chunks,
        "7. Panel d: coupled-field joint-distribution fidelity",
        "Base-2 Jensen–Shannon divergence (JSD), range 0–1, compares 64 × 64 joint histograms per state. Lower is better. The exact displayed pairs are **T–U1, CH4–U1, and p–U1**. Archived common truth-derived 0.5–99.5% histogram limits are used unchanged for every policy and model; histogram probabilities are normalized before the 1e−12 pseudocount is applied. The p–U1 edges are the transposed archived U1–p edges; JSD is unaffected by transposing both histograms.\n\n"
        + dual_matrix(policies, "joint_pdf_jsd_base2", pairs)
        + "\n\n"
        + dual_diffs(policies, "joint_pdf_jsd_base2", pairs)
        + "\n\n### Histogram-tail audit\n\nThe paper metric discards samples outside these fixed limits and renormalizes the remaining mass. The following percentages of reconstructed pairs remain inside the histogram; tail mass is not silently treated as faithful:\n\n"
        + dual_matrix(policies, "joint_pdf_reconstruction_retained_fraction", pairs, scale=100, decimals=3)
        + "\n\nTruth retained percentages (identical across checkpoints when truth arrays match): "
        + ", ".join(f"{target}: {100 * mean(primary, 'A0', 'joint_pdf_truth_retained_fraction', target):.3f}%" for target in pairs)
        + ".\n\n### Supplemental JSD retaining overflow values\n\nThe following is **not the original panel-d metric**. It extends the common 64-bin limits with underflow and overflow bins on each axis (66 × 66), so every finite point contributes. This tests whether apparent fidelity depends on discarding out-of-range predictions; lower is better.\n\n"
        + dual_matrix(policies, "joint_pdf_jsd_with_overflow_base2", pairs)
        + "\n\n"
        + dual_diffs(policies, "joint_pdf_jsd_with_overflow_base2", pairs)
        + "\n\nMean absolute error of the within-state Pearson correlation is an additional coarse coupling diagnostic (ideal 0):\n\n"
        + dual_matrix(policies, "coupling_correlation_abs_error", pairs)
        + "\n\nJoint-PDF agreement tests coupled statistical structure, not spatial alignment: a common spatial permutation of both predicted fields leaves their joint PDF unchanged. It therefore complements, rather than replaces, reconstruction L2.",
    )

    event_rows = []
    for policy in POLICIES:
        for _, item in nonpositive_events(policies[policy]).iterrows():
            event_rows.append([policy, item.method, int(item.snapshot), int(item.time_index), f"{float(item.value):.6f}"])
    count_text = (
        "not inferable"
        if not isinstance(context["n_points"], int)
        else f"{context['n_points'] * args.expected_snapshots:,}"
    )
    section(
        chunks,
        "8. Physical admissibility and pointwise structure",
        "Percent of predicted points with CH4 < 0, CO < 0, T ≤ 0, or p ≤ 0, evaluated before any clipping:\n\n"
        + dual_matrix(policies, "reconstruction_nonphysical_fraction", ["CH4", "CO", "T", "p"], scale=100, decimals=4)
        + "\n\nTruth fractions under the same check (%), shown from the primary truth arrays: "
        + ", ".join(f"{target}: {100 * mean(primary, 'A0', 'truth_nonphysical_fraction', target):.4f}" for target in ["CH4", "CO", "T", "p"])
        + ".\n\nTo distinguish tiny sign errors from material negative excursions, species percentages below **−0.0001** are also reported:\n\n"
        + dual_matrix(policies, "reconstruction_fraction_below_minus_1e4", ["CH4", "CO"], scale=100, decimals=4)
        + "\n\nThe negative part alone, normalized by the truth field norm, `||max(−prediction, 0)||₂ / ||truth||₂` (mean per state):\n\n"
        + dual_matrix(policies, "reconstruction_negative_l2_over_truth_l2", ["CH4", "CO"])
        + "\n\nMean magnitude of the negative part over all points (physical species units):\n\n"
        + dual_matrix(policies, "reconstruction_negative_mean_magnitude", ["CH4", "CO"], decimals=7)
        + f"\n\nTotal nonpositive-temperature/pressure points out of {count_text} evaluated points per model:\n\n"
        + nonphysical_count_table(policies, context["n_points"])
        + "\n\nIndividual states with nonpositive T (policy is explicit; rare events are not visible in a rounded average percentage):\n\n"
        + table(["Policy", "Run", "Test snapshot", "Original time index", "Minimum T"], event_rows)
        + "\n\nMinimum predicted physical value over all states/points (true summary minimum, not a mean of state minima):\n\n"
        + dual_extrema_matrix(policies, "reconstruction_minimum", ["CH4", "CO", "T", "p"], decimals=6)
        + "\n\nMean pointwise spatial Pearson correlation between each reconstructed field and its truth (ideal 1):\n\n"
        + dual_matrix(policies, "pointwise_correlation", FIELDS, decimals=5)
        + "\n\nThese checks do not establish PDE satisfaction, conservation, chemical source-term balance, or thermodynamic closure. Only one velocity component and two species are reconstructed here, so full continuity, momentum, species-sum, and energy-balance claims are not supported by this evaluation.",
    )

    section(
        chunks,
        "9. Stronger ablation: A5 versus A2",
        "This isolates the additional implemented removal of global observation-conditioning paths relative to A2, while retaining their respective saved checkpoint weights. It is not an equal-effective-capacity comparison.\n\n"
        + dual_diffs(policies, "physical_relative_l2", FIELDS + ["Unobserved_mean"], "A2")
        + "\n\n"
        + dual_diffs(policies, "spectral_lsd_db", ["CH4", "U1", "p"], "A2")
        + "\n\n"
        + dual_diffs(policies, "joint_pdf_jsd_base2", pairs, "A2"),
    )

    sensitivity_text = (
        f"Best and last per-state rows are paired by method, snapshot, time_index, metric, and target. The comparison uses {int(sensitivity.n_paired.max()) if not sensitivity.empty else 0:,} states per metric when complete; with a full run this is all {args.expected_snapshots:,} held-out states, rather than an outcome-selected subset. `best_minus_last` is the paired difference, and its uncertainty is a circular moving-block bootstrap of those per-state differences.\n\nPhysical relative-L2 checkpoint sensitivity:\n\n"
        + sensitivity_rows(sensitivity, "physical_relative_l2", FIELDS + ["Unobserved_mean"])
        + "\n\nNormalized relative-L2 checkpoint sensitivity:\n\n"
        + sensitivity_rows(sensitivity, "normalized_relative_l2", FIELDS + ["Unobserved_mean"])
        + "\n\nSpectral LSD checkpoint sensitivity:\n\n"
        + sensitivity_rows(sensitivity, "spectral_lsd_db", FIELDS, decimals=4)
        + "\n\nJoint-PDF and admissibility checkpoint sensitivity:\n\n"
        + sensitivity_rows(sensitivity, "joint_pdf_jsd_base2", pairs)
        + "\n\n"
        + sensitivity_rows(sensitivity, "reconstruction_nonphysical_fraction", ["CH4", "CO", "T", "p"], decimals=6)
        + f"\n\nAll metric/target pairs, including energy, overflow-JSD, retained-tail, negative-excursion, minima, and correlation diagnostics, are retained in [checkpoint_sensitivity_vs_last.csv]({sensitivity_path.resolve()}). This is a checkpoint-selection sensitivity analysis, not a basis for selecting a checkpoint from test outcomes."
    )
    section(chunks, "9b. Checkpoint-selection sensitivity", sensitivity_text)

    eq_rows = equivalence_rows(audit_map, policies)
    meta_notes = []
    for policy in POLICIES:
        metadata = policies[policy]["meta"]
        if metadata.get("truth_sensor_temporal_identity_matches") is not None:
            meta_notes.append(f"{policy} truth/sensor/time identity={metadata['truth_sensor_temporal_identity_matches']}")
    verification = read_json(root / "validation_summary.json")
    verification_text = ""
    if verification:
        verification_text = (
            f" The [completed validation summary]({root / 'validation_summary.json'}) records "
            f"{verification.get('regression_tests_passed', 'the')} passing regression tests, Python compilation, "
            "checkpoint-hash checks, complete coverage, and truth/sensor/time/seed identities across both policies."
        )
        for name in ("fresh_gpu1_validation_snapshot0.json", "cache_reuse_validation_best_snapshot0.json"):
            reuse_validation = root / name
            if reuse_validation.is_file():
                verification_text += f" Fresh inference versus reused arrays is documented in [cache reuse validation]({reuse_validation})."
                break
    validation_text = (
        table(
            ["Run", "Policy", "Legacy/cached equivalence passed", "Max normalized difference", "Relative-L2 difference", "Completed states", "Epoch"],
            eq_rows,
        )
        + "\n\nA1 compares its full forward path where the deterministic wrapper does not use streaming. Other small differences can arise from floating-point operation ordering. All truth arrays and sensor identities must match before paired claims; finite predictions and strict checkpoint loads are required by the evaluator. "
        + ("Recorded metadata: " + "; ".join(meta_notes) + ".\n\n" if meta_notes else "\n\n")
        + "The repository regression checks cover circular-block bootstrap behavior, snapshot-identity pairing, metric identity/scale, the archived sensor-plan/seed contract, and histogram-tail handling. Run `python -m unittest discover -s tests -p 'test_ablation_evaluation.py'` from the demo directory to reproduce those checks."
        + verification_text
        + "\n\n### Historical comparison with archived paper output (not validation of this evaluator)\n\n"
        + "The following compares the newly trained A0 per-state metric values with archived paper CSV values for historical context. It does not validate strict checkpoint loading, legacy/cached execution equivalence, truth/sensor pairing, or the new A0 implementation; those checks are listed above. The archived baseline is not substituted into any ablation table.\n\n"
        + historical_crosscheck_table(policies)
        + "\n\nHistogram assignments are discontinuous at bin boundaries, so small reconstruction differences can cause larger JSD changes than L2 changes. The crosscheck CSV records every comparison, not merely rounded panel means.\n\nThere is one trained checkpoint per variant, one inference seed per state, and one sensor plan. No multi-seed training significance, generative calibration, sample diversity, solver-convergence, or independent-test generalization claim is made."
    )
    section(chunks, "10. Validation and numerical reproducibility", validation_text)

    worst_rows = []
    for policy in POLICIES:
        for _, item in worst_states(policies[policy]).iterrows():
            worst_rows.append([policy, item.method, int(item.snapshot), int(item.time_index), f"{float(item.value):.5f}"])
    section(
        chunks,
        "11. Most difficult states",
        table(["Policy", "Run", "Test snapshot index", "Original simulation time index", "Unobserved mean L2"], worst_rows)
        + "\n\nThe complete per-state tables support further inspection of any field, pair, or tail event without repeating inference.",
    )

    artifact_lines = []
    for policy in POLICIES:
        metrics_dir = policies[policy]["dir"]
        for name, description in [
            ("summary_metrics.csv", "metric means, dispersion, quantiles, and bootstrap intervals"),
            ("per_state_metrics.csv", "all per-state values, checkpoint policy, time identities, and cache paths"),
            ("paired_differences.csv", "paired ablation changes and block-5/20/50 intervals"),
            ("cache_audit.csv", "truth/sensor hashes and temporal identity checks"),
            ("A0_paper_crosscheck.csv", "historical A0 comparison with archived paper values"),
            ("analysis_metadata.json", "coverage, metric-code hashes, histogram-edge hashes, and statistical settings"),
            ("snapshot0_spectra.csv", "full shell spectra for a reproducible example state"),
        ]:
            path = metrics_dir / name
            if path.is_file():
                artifact_lines.append(f"- [{policy}/{name}]({path.resolve()}): {description}")
    artifact_lines.append(f"- [checkpoint_sensitivity_vs_last.csv]({sensitivity_path.resolve()}): all-state paired best-minus-last estimates and block-bootstrap uncertainty")
    environment_path = root / "evaluation_environment.json"
    python_command = "python"
    inference_manifest = read_json(root / "inference_manifest.json")
    reproduction_commands = []
    for policy in POLICIES:
        recorded = inference_manifest.get("reproduction_commands", {}).get(policy)
        if recorded:
            # Recorded commands may include a working-directory prefix. Use
            # absolute script paths in the report so they run from repo root.
            recorded = recorded.split(" && ")[-1]
            recorded = recorded.replace(" src/evaluate_ablation_condT.py", f" {DEMO}/src/evaluate_ablation_condT.py")
            reproduction_commands.append(recorded)
        else:
            a0_option = f" --a0-run-dir {a0_input.resolve()}" if a0_input.is_dir() else ""
            reproduction_commands.append(
                f"{python_command} {DEMO}/src/evaluate_ablation_condT.py --evaluation-dir {root.resolve()}"
                f" --checkpoint {policy} --device {device}{a0_option} --verify-equivalence --uncompressed"
            )
        reproduction_commands.append(
            f"{python_command} {DEMO}/src/analyze_ablation_condT.py --evaluation-dir {root.resolve()}/{policy}"
            f" --workers 4 --expected-snapshots {args.expected_snapshots}"
        )
    if (root / "high_frequency" / "analysis_metadata.json").is_file():
        figure_python = "python"
        reproduction_commands.append(
            f"{figure_python} {DEMO}/src/analyze_ablation_high_frequency.py"
            f" --evaluation-dir {root.resolve()} --expected-snapshots {args.expected_snapshots}"
            " --workers 8 --bootstrap-resamples 2000"
        )
        reproduction_commands.append(
            f"{figure_python} {DEMO}/figures/scripts/plot_ablation_high_frequency.py"
            f" --input-dir {root.resolve()}/high_frequency"
            f" --output-dir {DEMO}/figures/generated/ablation_condT_high_frequency_{root.name.removeprefix('evaluation_')}"
        )
    reproduction_commands.append(f"{python_command} {Path(__file__).resolve()} --evaluation-dir {root.resolve()} --output {output}")
    section(
        chunks,
        "12. Reproduction and deliverables",
        "From the repository root (using the existing environment):\n\n```bash\n"
        + "\n".join(reproduction_commands) + "\n```\n\n"
        "Inference resumes only compatible cached states and refuses incompatible checkpoint/config/sensor-plan/code identities. The report command regenerates both-policy quantitative tables and embeds `reviewed_interpretation.md` from the evaluation root when present; the reviewed narrative remains a separate source document.\n\n"
        + "\n".join(artifact_lines)
        + f"\n\nCheckpoint provenance: [{root.name}]({root.resolve()}); each policy's `provenance_A*.json` and `manifest_A*.csv` links exact checkpoints, configurations, data, generation seeds, and reconstructions. "
        + (f"[evaluation_environment.json]({environment_path.resolve()}) records Git revision, environment versions, model/data-source hashes, device, and cache-storage policy.\n\n" if environment_path.is_file() else "The evaluation environment metadata file was not present at report generation.\n\n")
        + "The original checkpoints, archived figure/results, and unrelated workspace revisions were not modified by report generation. "
        + ("The new high-frequency diagnostic figure and its editable exports are linked in Section 6b. " if spectral_supplement.is_file() else "")
        + "The detailed Markdown and reusable evaluation artifacts are the deliverables.",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

"""Strict source adapters for the additive Figure 5 V4 workflow.

Panels a--c intentionally reuse the adopted, QA-passing V3 products.  Panels d
and e have independent V4 roots and schemas: a V3 query table is never accepted
as training-compute or high-resolution stress evidence.  The adapters return
``None`` for incomplete sources so interactive builds can render an explicit
pending panel; strict builds turn the same diagnostics into a hard failure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceRecord:
    """Provenance/status row written to the V4 build manifest."""

    panel: str
    mode: str
    status: str
    source: str
    note: str


ALLOWED_FAILURE_STATUSES = {
    "oom",
    "cuda_oom",
    "runtime_cap",
    "memory_cap",
    "safety_cap",
    "failed",
}


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(table: pd.DataFrame, columns: list[str]) -> bool:
    if any(column not in table.columns for column in columns):
        return False
    try:
        values = table[columns].to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(values).all())


def _bool_series(values: pd.Series) -> pd.Series:
    """Parse CSV booleans without treating ``"False"`` as true."""

    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False}
    ).fillna(False).astype(bool)


def _manifest_metric(manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return metric name/unit from either supported manifest spelling."""

    metric = manifest.get("metric")
    if isinstance(metric, dict):
        name = metric.get("name")
        unit = metric.get("unit")
    else:
        name = metric
        unit = None
    name = manifest.get("metric_name") or name
    unit = manifest.get("metric_unit") or unit or manifest.get("unit")
    return (None if name is None else str(name), None if unit is None else str(unit))


def _formal_run(
    root: Path,
    run_id: str,
    schema: str,
    required: list[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Resolve a complete manifest/QA-gated run without considering fallbacks."""

    directory = root / run_id
    errors: list[str] = []
    if not directory.is_dir():
        return None, [f"missing run directory: {directory}"]
    manifest_path, qa_path = directory / "manifest.json", directory / "qa.json"
    if not manifest_path.is_file():
        errors.append("missing manifest.json")
    if not qa_path.is_file():
        errors.append("missing qa.json")
    if errors:
        return None, errors
    try:
        manifest, qa = _json(manifest_path), _json(qa_path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid manifest/QA JSON: {exc}"]
    if manifest.get("schema_version") != schema:
        errors.append(f"schema_version={manifest.get('schema_version')!r}, expected {schema!r}")
    if manifest.get("status") != "complete":
        errors.append(f"manifest status={manifest.get('status')!r}")
    if manifest.get("formal") is not True:
        errors.append("manifest formal flag is not true")
    if qa.get("status") != "pass":
        errors.append(f"QA status={qa.get('status')!r}")
    for name in required:
        if not (directory / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return None, errors
    return {"directory": directory, "manifest": manifest, "qa": qa}, []


def _load_v3_uq(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    formal = config["formal_inputs"]
    root = _repo_path(repo_root, formal["uq_root"])
    run, errors = _formal_run(
        root,
        str(formal["uq_run_id"]),
        "figure5-validation-v3-uq-1",
        ["crps_summary.csv", "spread_error_summary.csv", "per_state_method.csv", "reliability_si.csv"],
    )
    if run is None:
        return None, errors
    if "ValidationV2" in str(run["directory"]):
        return None, ["V2 uncertainty source is not admissible for V4"]
    try:
        crps = pd.read_csv(run["directory"] / "crps_summary.csv")
        spread = pd.read_csv(run["directory"] / "spread_error_summary.csv")
        states = pd.read_csv(run["directory"] / "per_state_method.csv")
    except (OSError, pd.errors.ParserError) as exc:
        return None, [f"could not read V3 UQ tables: {exc}"]
    methods = list(config["paper_contract"]["generative_method_order"])
    if list(crps.get("method", pd.Series(dtype=str)).astype(str)) != methods:
        errors.append("CRPS method order does not match the V4 generative order")
    if list(spread.get("method", pd.Series(dtype=str)).astype(str)) != methods:
        errors.append("spread/error method order does not match the V4 generative order")
    if len(states) != len(methods) * 200 or states.get("state", pd.Series(dtype=int)).nunique() != 200:
        errors.append("V3 UQ state/method cohort is not 200 paired states")
    if "draw_count" not in states.columns or set(states["draw_count"].astype(int)) != {64}:
        errors.append("V3 UQ draw count is not exactly 64")
    if "method" in states.columns and any(states[states["method"].eq(method)]["state"].nunique() != 200 for method in methods):
        errors.append("V3 UQ methods do not share the paired state cohort")
    if not _finite(crps, ["mean_normalized_crps", "crps_ci_low", "crps_ci_high"]):
        errors.append("V3 CRPS table contains missing/non-finite estimates")
    if not _finite(spread, ["spearman_rho", "spearman_ci_low", "spearman_ci_high"]):
        errors.append("V3 spread/error table contains missing/non-finite estimates")
    if errors:
        return None, errors
    run.update({"crps": crps, "spread": spread, "states": states})
    return run, []


def _load_v3_native(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    formal = config["formal_inputs"]
    root = _repo_path(repo_root, formal["cost_root"])
    run, errors = _formal_run(
        root,
        str(formal["cost_run_id"]),
        "figure5-validation-v3-cost-1",
        ["native_summary.csv", "timing_boundary_audit.csv"],
    )
    if run is None:
        return None, errors
    if "ValidationV2" in str(run["directory"]):
        return None, ["V2 cost source is not admissible for V4"]
    try:
        native = pd.read_csv(run["directory"] / "native_summary.csv")
    except (OSError, pd.errors.ParserError) as exc:
        return None, [f"could not read V3 native-cost table: {exc}"]
    methods = list(config["paper_contract"]["method_order"])
    if list(native.get("method", pd.Series(dtype=str)).astype(str)) != methods:
        errors.append("V3 native-cost method order does not match the V4 method order")
    if set(native.get("status", pd.Series(dtype=str)).astype(str)) != {"ok"}:
        errors.append("V3 native-cost rows are not all status=ok")
    if "N" not in native.columns or set(native["N"].astype(int)) != {int(config["paper_contract"]["native_query_count"])}:
        errors.append("V3 native-cost rows are not all at N=40,300")
    if not _finite(native, ["median_latency_ms", "latency_q25_ms", "latency_q75_ms", "error", "error_ci_low", "error_ci_high"]):
        errors.append("V3 native-cost table contains missing/non-finite estimates")
    if errors:
        return None, errors
    native = native.copy()
    native["cost_value"] = native["median_latency_ms"].astype(float)
    native["cost_low"] = native["latency_q25_ms"].astype(float)
    native["cost_high"] = native["latency_q75_ms"].astype(float)
    native["cost_metric"] = "warm_model_core_latency_ms"
    native["cost_unit"] = "ms"
    run["native"] = native
    return run, []


def _normalise_training_table(table: pd.DataFrame, manifest: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame | None, list[str]]:
    errors: list[str] = []
    aliases = {
        "cost_value": ("cost_value", "training_cost_value", "training_cost_gpu_hours", "gpu_hours", "replay_equivalent_gpu_hours", "training_update_time_ms"),
        "cost_low": ("cost_low", "training_cost_low", "gpu_hours_low", "update_time_low_ms"),
        "cost_high": ("cost_high", "training_cost_high", "gpu_hours_high", "update_time_high_ms"),
    }
    table = table.copy()
    for target, candidates in aliases.items():
        if target not in table.columns:
            source = next((candidate for candidate in candidates if candidate in table.columns), None)
            if source is not None:
                table[target] = table[source]
    required = ["method", "status", "cost_value", "cost_low", "cost_high", "error", "error_ci_low", "error_ci_high"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        return None, [f"training-cost table missing columns: {', '.join(missing)}"]
    manifest_metric, manifest_unit = _manifest_metric(manifest)
    metric = str(
        manifest_metric
        or config["formal_protocol"]["training_cost"]["preferred_metric"]
    )
    allowed = set(config["formal_protocol"]["training_cost"]["allowed_metrics"])
    if metric not in allowed:
        errors.append(f"unsupported training-cost metric {metric!r}; allowed={sorted(allowed)}")
    table["method"] = table["method"].astype(str)
    methods = list(config["paper_contract"]["method_order"])
    unknown = sorted(set(table["method"]) - set(methods))
    if unknown:
        errors.append(f"training-cost table contains unknown methods: {unknown}")
    ok = table["status"].astype(str).str.lower().eq("ok")
    if not bool(ok.any()):
        errors.append("training-cost table has no status=ok row")
    if bool(ok.any()) and not _finite(table.loc[ok], ["cost_value", "cost_low", "cost_high", "error", "error_ci_low", "error_ci_high"]):
        errors.append("valid training-cost rows contain missing/non-finite values")
    if bool(ok.any()) and (table.loc[ok, "cost_value"].astype(float) <= 0).any():
        errors.append("valid training-cost values must be positive")
    if errors:
        return None, errors
    table["cost_value"] = table["cost_value"].astype(float)
    table["cost_low"] = table["cost_low"].astype(float)
    table["cost_high"] = table["cost_high"].astype(float)
    table["cost_metric"] = metric
    table["cost_unit"] = str(manifest_unit or ("ms/update" if metric == "training_update_time_ms" else "GPU h"))
    table["training_cost_basis"] = table.get("training_cost_basis", manifest.get("basis", metric))
    return table, []


def _load_training_cost(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    formal = config["formal_inputs"]
    root = _repo_path(repo_root, formal["training_cost_root"])
    run, errors = _formal_run(
        root,
        str(formal["training_cost_run_id"]),
        "figure5-validation-v4-training-cost-1",
        ["training_cost_summary.csv"],
    )
    if run is None:
        return None, errors
    try:
        table = pd.read_csv(run["directory"] / "training_cost_summary.csv")
    except (OSError, pd.errors.ParserError) as exc:
        return None, [f"could not read V4 training-cost table: {exc}"]
    table, table_errors = _normalise_training_table(table, run["manifest"], config)
    if table_errors:
        return None, table_errors
    # A replay-equivalent estimate must explicitly document its validation gate;
    # otherwise a plausible-looking number is not promoted as panel-d evidence.
    metric = str(table["cost_metric"].iloc[0])
    if metric == "replay_equivalent_gpu_hours":
        gate = run["manifest"].get("promotion_gate", {})
        tolerance = float(config["formal_protocol"]["training_cost"]["promotion_tolerance_fraction"])
        if gate.get("validated") is not True or float(gate.get("tolerance_fraction", tolerance + 1.0)) > tolerance:
            return None, ["replay-equivalent GPU-hours lack a passing predeclared validation gate"]
    run["training"] = table
    run["metric_name"] = metric
    run["metric_label"] = str(run["manifest"].get("metric_label") or ("Training update time (ms/update)" if metric == "training_update_time_ms" else f"{metric.replace('_', ' ').title()}"))
    return run, []


def _load_v3_scaling_native(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Load the validated V3 native query/memory prefix for panel e.

    V3 owns the physically validated ``N={1024,4096,16384,40300}`` rows.  The
    V4 stress runner is never allowed to replace those rows or to provide a
    native-region fallback; this helper only joins the exact V3 run after its
    manifest and query/memory QA gates pass.
    """

    formal = config["formal_inputs"]
    root = _repo_path(repo_root, formal["cost_root"])
    run, errors = _formal_run(
        root,
        str(formal["cost_run_id"]),
        "figure5-validation-v3-cost-1",
        ["query_latency_summary.csv", "memory_summary.csv", "variable_query_support.csv"],
    )
    if run is None:
        return None, errors
    if "ValidationV2" in str(run["directory"]):
        return None, ["V2 query/memory source is not admissible for V4"]
    try:
        latency = pd.read_csv(run["directory"] / "query_latency_summary.csv")
        memory = pd.read_csv(run["directory"] / "memory_summary.csv")
        support = pd.read_csv(run["directory"] / "variable_query_support.csv")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return None, [f"could not read V3 native query/memory tables: {exc}"]

    methods = list(config["paper_contract"]["method_order"])
    native_counts = [int(value) for value in config["formal_protocol"]["scale_stress"]["native_query_counts"]]
    native = int(config["formal_protocol"]["scale_stress"]["native_limit"])
    if list(support.get("method", pd.Series(dtype=str)).astype(str)) != methods:
        errors.append("V3 variable-query support order does not match the V4 method order")
    if "variable_query_supported" not in support:
        errors.append("V3 variable-query support lacks variable_query_supported")
        variable_methods: set[str] = set()
    else:
        support = support.copy()
        support["variable_query_supported"] = _bool_series(support["variable_query_supported"])
        if "native_only" in support:
            support["native_only"] = _bool_series(support["native_only"])
        variable_methods = set(support.loc[support["variable_query_supported"], "method"].astype(str))
    if not variable_methods:
        errors.append("V3 native source has no variable-query methods")

    for table_name, table, value_columns in (
        ("query latency", latency, ["median_latency_ms", "latency_q25_ms", "latency_q75_ms"]),
        ("memory", memory, ["peak_allocated_mib"]),
    ):
        missing = {"method", "N", "status"} - set(table.columns)
        if missing:
            errors.append(f"V3 {table_name} table missing columns: {sorted(missing)}")
            continue
        table["method"] = table["method"].astype(str)
        try:
            table["N"] = pd.to_numeric(table["N"], errors="raise").astype(int)
        except (TypeError, ValueError):
            errors.append(f"V3 {table_name} table has non-integer N")
            continue
        if set(table["method"]) != set(methods):
            errors.append(f"V3 {table_name} table method set does not match the V4 method order")
        if set(table["status"].astype(str).str.lower()) != {"ok"}:
            errors.append(f"V3 {table_name} table is not all status=ok")
        if not _finite(table, value_columns):
            errors.append(f"V3 {table_name} table contains non-finite native values")
        for method in methods:
            expected = set(native_counts if method in variable_methods else [native])
            observed = set(table.loc[table["method"].eq(method), "N"].astype(int))
            if observed != expected:
                errors.append(f"V3 {table_name} rows for {method} do not match native grid {sorted(expected)}")

    if (
        {"method", "N"}.issubset(latency.columns)
        and {"method", "N"}.issubset(memory.columns)
        and set(zip(latency["method"], latency["N"])) != set(zip(memory["method"], memory["N"]))
    ):
        errors.append("V3 query latency/memory keys differ")
    qa = run["qa"]
    required_qa = {
        "query_memory_protocol_match": True,
        "no_full_grid_then_slice_scaling": True,
        "timing_protocol_pass": True,
        "identity_pass": True,
        "gpu_clean_before": True,
        "gpu_clean_after": True,
    }
    for key, expected in required_qa.items():
        if qa.get(key) is not expected:
            errors.append(f"V3 native QA does not pass {key}")
    protocol = run["manifest"].get("protocol", {})
    if [int(value) for value in protocol.get("query_counts", [])] != native_counts:
        errors.append("V3 native manifest query_counts do not match the V4 native grid")
    if protocol.get("throughput_extension") not in {"not_run", None}:
        errors.append("V3 source includes an unapproved throughput extension")
    if errors:
        return None, errors

    identity_by_method: dict[str, tuple[str, str]] = {}
    for method in methods:
        rows = latency.loc[(latency["method"].eq(method)) & (latency["N"].eq(native))]
        row = rows.iloc[0]
        identity_by_method[method] = (str(row.get("checkpoint_path", "")), str(row.get("checkpoint_sha256", "")))

    def canonical(table: pd.DataFrame, *, metric: str) -> pd.DataFrame:
        output = table.copy()
        output["query_region"] = "native_validated"
        output["throughput_only"] = False
        output["accuracy_claim"] = True
        output["source_schema"] = "figure5-validation-v3-cost-1"
        output["query_spec_hash"] = "v3-native-validated-source"
        output["variable_query_supported"] = output["method"].map(lambda method: method in variable_methods)
        output["native_only"] = ~output["variable_query_supported"]
        output["cost_metric"] = metric
        output["checkpoint_path"] = output["method"].map(lambda method: identity_by_method[method][0])
        output["checkpoint_sha256"] = output["method"].map(lambda method: identity_by_method[method][1])
        return output

    return {
        "directory": run["directory"],
        "manifest": run["manifest"],
        "qa": run["qa"],
        "latency": canonical(latency, metric="warm_model_core_latency_ms"),
        "memory": canonical(memory, metric="peak_allocated_mib"),
        "support": support,
    }, []


def _prefix_counts(values: list[int], declared: list[int]) -> bool:
    """Return whether attempted counts are a contiguous declared prefix."""

    return values == declared[: len(values)]


def _load_scale_stress(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    formal = config["formal_inputs"]
    # The stress runner owns only N>40,300.  Validate and retain the V3 native
    # prefix separately so a V4 run cannot accidentally replace physically
    # validated rows with dummy-query measurements.
    v3_native, v3_errors = _load_v3_scaling_native(config, repo_root)
    if v3_native is None:
        return None, ["V4 scale run requires the validated V3 native query/memory prefix."] + v3_errors

    root = _repo_path(repo_root, formal["scale_root"])
    run, errors = _formal_run(
        root,
        str(formal["scale_run_id"]),
        "figure5-validation-v4-scale-stress-1",
        [
            "scale_stress_summary.csv",
            "native_query_support_audit.csv",
            "boundary_summary.csv",
            "query_coordinates_manifest.csv",
        ],
    )
    if run is None:
        return None, errors
    directory, manifest, qa = run["directory"], run["manifest"], run["qa"]
    try:
        stress = pd.read_csv(directory / "scale_stress_summary.csv")
        support = pd.read_csv(directory / "native_query_support_audit.csv")
        boundary = pd.read_csv(directory / "boundary_summary.csv")
        queries = pd.read_csv(directory / "query_coordinates_manifest.csv")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return None, [f"could not read V4 scale-stress tables: {exc}"]
    methods = list(config["paper_contract"]["method_order"])
    native = int(config["formal_protocol"]["scale_stress"]["native_limit"])
    scale_cfg = config["formal_protocol"]["scale_stress"]
    native_counts = [int(value) for value in scale_cfg["native_query_counts"]]
    stress_counts = [int(value) for value in scale_cfg["throughput_query_counts"]]
    adaptive_cap = int(scale_cfg.get("adaptive_query_cap", stress_counts[-1]))
    candidate_counts = list(stress_counts)
    while candidate_counts[-1] < adaptive_cap:
        candidate_counts.append(min(adaptive_cap, candidate_counts[-1] * 2))
    if native_counts[-1] != native:
        errors.append("V4 native grid must end at the native query limit")
    if native_counts != [1024, 4096, 16384, native]:
        errors.append("V4 native query grid is not the frozen 1,024/4,096/16,384/40,300 grid")
    if not stress_counts or stress_counts[0] <= native or stress_counts != [100000, 250000, 500000, 1000000, 2000000, 4000000]:
        errors.append("V4 throughput grid is not the frozen 100k..4M predeclared grid")
    if adaptive_cap != 8000000:
        errors.append("V4 adaptive query cap is not the frozen 8M cap")

    # The architecture audit is authoritative for V4 eligibility.  The V3
    # support table must agree, but it does not provide a V4 query hash.
    support = support.copy()
    support["method"] = support.get("method", pd.Series(dtype=str)).astype(str)
    if list(support["method"]) != methods:
        errors.append("V4 native-query support method order does not match the V4 method order")
    for column in ("native_query_supported", "query_scaling_eligible", "native_only"):
        if column not in support.columns:
            errors.append(f"V4 native-query support lacks {column}")
            support[column] = False
        else:
            support[column] = _bool_series(support[column])
    if "status" not in support.columns or set(support["status"].astype(str).str.lower()) != {"ok"}:
        errors.append("V4 native-query support audit is not complete/status=ok for every method")
    eligible_methods = set(
        support.loc[
            support.get("native_query_supported", pd.Series(False, index=support.index))
            & support.get("query_scaling_eligible", pd.Series(False, index=support.index)),
            "method",
        ].astype(str)
    )
    variable_methods = [method for method in methods if method in eligible_methods]
    variable_method_set = set(variable_methods)
    # Normalize the canonical V4 architecture-audit decision to the plotting
    # interface shared with the validated V3 native prefix.  Keep the original
    # audit columns intact so the derived source table retains the exact basis
    # for each eligibility decision.
    support["variable_query_supported"] = support["method"].isin(variable_methods)
    if not variable_methods:
        errors.append("V4 scale evidence has no canonical variable-query method")
    if set(support.loc[support["native_only"], "method"].astype(str)) != set(methods) - variable_method_set:
        errors.append("V4 native-query support native_only flags do not match eligible methods")
    v3_variable = set(v3_native["support"].loc[v3_native["support"]["variable_query_supported"], "method"].astype(str))
    if v3_variable != variable_method_set:
        errors.append("V4 architecture audit disagrees with the validated V3 query-evaluable method set")
    if "decision_basis" not in support.columns or support["decision_basis"].astype(str).str.strip().eq("").any():
        errors.append("V4 support audit lacks a decision basis for one or more methods")
    forbidden = ("full_grid", "full grid", "slice", "slicing", "reconstruct_then")
    if "decision_basis" in support.columns:
        for row in support[support["method"].isin(variable_methods)].itertuples():
            if any(token in str(row.decision_basis).lower() for token in forbidden):
                errors.append(f"variable-query basis for {row.method} indicates full-grid/slicing")

    # The runner stores the common query hash in the manifest/query manifest,
    # not in each architecture-audit row.
    protocol = manifest.get("protocol", {})
    dummy_spec = manifest.get("dummy_query_spec", {})
    declared_spec_hash = str(manifest.get("dummy_query_spec_sha256") or "")
    if not declared_spec_hash:
        errors.append("V4 manifest lacks dummy_query_spec_sha256")
    if not isinstance(dummy_spec, dict) or dummy_spec.get("generator") != "torch.quasirandom.SobolEngine":
        errors.append("V4 query specification is not the canonical Sobol generator")
    if not isinstance(dummy_spec, dict) or dummy_spec.get("include_sensor_prefix") is not True or dummy_spec.get("sequence_policy") != "exact_sensor_prefix_then_sobol_suffix":
        errors.append("V4 query specification is not sensor-prefixed Sobol")
    if protocol.get("native_query_count") != native or protocol.get("sensor_count") != int(config["paper_contract"]["sensor_count"]):
        errors.append("V4 scale manifest native/sensor protocol does not match the paper contract")
    if [int(value) for value in protocol.get("predeclared_query_counts", [])] != stress_counts:
        errors.append("V4 manifest predeclared query counts do not match the frozen stress grid")
    if [int(value) for value in protocol.get("candidate_query_counts", [])] != candidate_counts:
        errors.append("V4 manifest candidate query counts do not match the frozen adaptive cap")
    if int(protocol.get("global_query_cap", -1)) != adaptive_cap:
        errors.append("V4 manifest global query cap does not match the config")

    query_required = {"N", "query_sha256", "spec_sha256", "throughput_only", "accuracy_claim"}
    if not query_required.issubset(queries.columns):
        errors.append(f"query_coordinates_manifest.csv missing columns: {sorted(query_required - set(queries.columns))}")
        query_counts: list[int] = []
        query_hashes: dict[int, str] = {}
    else:
        queries = queries.copy()
        queries["N"] = pd.to_numeric(queries["N"], errors="coerce")
        if queries["N"].isna().any():
            errors.append("query coordinate manifest has non-integer N")
        queries["N"] = queries["N"].fillna(-1).astype(int)
        queries["throughput_only"] = _bool_series(queries["throughput_only"])
        queries["accuracy_claim"] = _bool_series(queries["accuracy_claim"])
        query_counts = sorted(set(queries["N"]))
        query_hashes = {}
        for count, group in queries.groupby("N"):
            hashes = set(group["query_sha256"].astype(str))
            specs = set(group["spec_sha256"].astype(str))
            if len(hashes) != 1 or len(specs) != 1 or (declared_spec_hash and specs != {declared_spec_hash}):
                errors.append(f"query coordinate hashes are inconsistent at N={count}")
            query_hashes[int(count)] = next(iter(hashes)) if len(hashes) == 1 else ""
        if query_counts and any(count <= native or count not in candidate_counts for count in query_counts):
            errors.append("V4 query coordinate manifest includes a non-throughput or undeclared N")
        if not queries["throughput_only"].all() or queries["accuracy_claim"].any():
            errors.append("V4 query coordinate manifest does not carry the throughput-only/no-accuracy flags")
        if "generator" in queries and set(queries["generator"].astype(str)) != {"torch.quasirandom.SobolEngine"}:
            errors.append("V4 query coordinate manifest does not record Sobol generation")
        if "sensor_count" in queries and set(pd.to_numeric(queries["sensor_count"], errors="coerce")) != {int(config["paper_contract"]["sensor_count"])}:
            errors.append("V4 query coordinate manifest sensor prefix is not 256 points")

    # The stress runner emits one combined row carrying latency and memory.
    summary_required = {
        "method", "N", "status", "median_latency_ms", "latency_q25_ms", "latency_q75_ms",
        "peak_allocated_mib", "query_sha256", "throughput_only", "accuracy_claim",
    }
    if not summary_required.issubset(stress.columns):
        errors.append(f"scale_stress_summary.csv missing columns: {sorted(summary_required - set(stress.columns))}")
        # Do not continue into row-level checks with a malformed table: strict
        # callers need a diagnostic, not an adapter traceback.
        return None, errors
    else:
        stress = stress.copy()
        stress["method"] = stress["method"].astype(str)
        stress["N"] = pd.to_numeric(stress["N"], errors="coerce")
        if stress["N"].isna().any():
            errors.append("scale stress summary has non-integer or unavailable N rows")
        stress["N"] = stress["N"].fillna(-1).astype(int)
        stress["throughput_only"] = _bool_series(stress["throughput_only"])
        stress["accuracy_claim"] = _bool_series(stress["accuracy_claim"])
    if not stress.empty:
        unknown = sorted(set(stress["method"]) - variable_method_set)
        if unknown:
            errors.append(f"scale stress summary contains unsupported methods: {unknown}")
        if (stress["N"] <= native).any() or not set(stress["N"]).issubset(set(candidate_counts)):
            errors.append("scale stress summary includes a non-throughput or undeclared query count")
        if not stress["throughput_only"].all() or stress["accuracy_claim"].any():
            errors.append("scale stress summary does not carry the throughput-only/no-accuracy flags")
        status_values = stress["status"].astype(str).str.lower()
        allowed_status = {"ok", "boundary_failure", "failed", "oom", "cuda_oom", "runtime_cap", "memory_cap", "safety_cap"}
        if set(status_values) - allowed_status:
            errors.append(f"scale stress summary has unsupported statuses: {sorted(set(status_values) - allowed_status)}")
        for method in variable_methods:
            rows = stress[stress["method"].eq(method)].sort_values("N")
            counts = [int(value) for value in rows["N"]]
            if not _prefix_counts(counts, candidate_counts):
                errors.append(f"scale stress attempts for {method} are not a declared prefix")
            statuses = rows["status"].astype(str).str.lower().tolist()
            failure_positions = [index for index, status in enumerate(statuses) if status != "ok"]
            if failure_positions and failure_positions[-1] != len(statuses) - 1:
                errors.append(f"scale stress rows for {method} continue after the first failure")
            valid = rows[rows["status"].astype(str).str.lower().eq("ok")]
            if not valid.empty:
                if not _finite(valid, ["median_latency_ms", "latency_q25_ms", "latency_q75_ms", "peak_allocated_mib"]):
                    errors.append(f"scale stress valid rows for {method} contain non-finite metrics")
                if (valid[["median_latency_ms", "latency_q25_ms", "latency_q75_ms", "peak_allocated_mib"]].astype(float) <= 0).any().any():
                    errors.append(f"scale stress valid rows for {method} contain non-positive metrics")
                if (valid["latency_q25_ms"].astype(float) > valid["median_latency_ms"].astype(float)).any() or (valid["median_latency_ms"].astype(float) > valid["latency_q75_ms"].astype(float)).any():
                    errors.append(f"scale stress latency quartiles are not ordered for {method}")
            for row in rows.itertuples():
                if int(row.N) not in query_hashes or str(row.query_sha256) != query_hashes[int(row.N)]:
                    errors.append(f"scale stress query hash does not match the shared coordinate manifest at N={int(row.N)}")

    attempted_stress_counts = sorted(set(stress["N"].astype(int))) if not stress.empty and "N" in stress else []
    if query_counts != attempted_stress_counts:
        errors.append("query coordinate manifest counts do not equal the union of attempted stress rows")

    boundary_required = {"method", "largest_success_N", "first_failure_N", "termination_reason"}
    if not boundary_required.issubset(boundary.columns):
        errors.append(f"boundary_summary.csv missing columns: {sorted(boundary_required - set(boundary.columns))}")
        canonical_boundary = pd.DataFrame(columns=list(boundary_required))
    else:
        boundary = boundary.copy()
        boundary["method"] = boundary["method"].astype(str)
        if set(boundary["method"]) != variable_method_set or len(boundary) != len(variable_methods):
            errors.append("boundary summary does not contain exactly one row per variable-query method")
        canonical_rows: list[dict[str, Any]] = []
        for method in variable_methods:
            rows = stress[stress["method"].eq(method)] if "method" in stress else pd.DataFrame()
            status_values = rows["status"].astype(str).str.lower() if not rows.empty else pd.Series(dtype=str)
            ok_counts = [int(value) for value in rows.loc[status_values.eq("ok"), "N"]]
            failure_counts = [int(value) for value in rows.loc[~status_values.eq("ok"), "N"]]
            source = boundary[boundary["method"].eq(method)]
            if len(source) != 1:
                errors.append(f"boundary summary has no unique row for {method}")
                continue
            source_row = source.iloc[0]
            largest_value = pd.to_numeric(pd.Series([source_row["largest_success_N"]]), errors="coerce").iloc[0]
            first_value = pd.to_numeric(pd.Series([source_row["first_failure_N"]]), errors="coerce").iloc[0]
            source_largest = None if pd.isna(largest_value) else int(largest_value)
            source_first = None if pd.isna(first_value) else int(first_value)
            expected_largest = max(ok_counts) if ok_counts else None
            expected_first = min(failure_counts) if failure_counts else None
            if source_largest != expected_largest:
                errors.append(f"boundary largest_success_N disagrees with stress rows for {method}")
            if source_first != expected_first:
                errors.append(f"boundary first_failure_N disagrees with stress rows for {method}")
            reason = str(source_row["termination_reason"])
            if expected_first is not None and reason != "first_failure":
                errors.append(f"boundary termination reason for {method} is not first_failure")
            if expected_first is None and (expected_largest != candidate_counts[-1] or reason != "global_cap_reached"):
                errors.append(f"boundary termination reason for {method} is not global-cap completion")
            # The plotted endpoint includes the V3 native prefix even when the
            # first V4 stress request fails before one successful stress row.
            canonical_rows.append({
                "method": method,
                "largest_successful_N": max(native, source_largest or native),
                "first_failed_N": source_first,
                "termination_reason": reason,
            })
        canonical_boundary = pd.DataFrame(canonical_rows)

    # The runner's QA deliberately records boundary rows; all-success is not a
    # requirement because a prefix ending at a declared hardware boundary is
    # valid evidence.
    for key in (
        "support_methods_exact", "all_eligible_attempted", "candidate_counts_predeclared",
        "shared_query_hash_per_count", "largest_success_first_failure_recorded",
        "latency_iqr_valid", "throughput_only_no_accuracy_claim",
        "no_unsupported_scaling_curve", "geometry_preparation_separate",
        "repeat_rows_present", "identity_pass", "gpu_clean_before", "gpu_clean_after",
        "fixed_grid_methods_have_no_scaling_curve",
    ):
        if qa.get(key) is not True:
            errors.append(f"V4 scale QA does not pass {key}")
    if qa.get("status") != "pass":
        errors.append(f"V4 scale QA status={qa.get('status')!r}")
    if manifest.get("no_accuracy_claim_above_native") is not None and manifest.get("no_accuracy_claim_above_native") is not True:
        errors.append("V4 manifest does not prohibit accuracy claims above native N")
    if qa.get("no_accuracy_claim_above_native") is not None and qa.get("no_accuracy_claim_above_native") is not True:
        errors.append("V4 QA does not prohibit accuracy claims above native N")

    if errors:
        return None, errors

    stress["query_region"] = "throughput_only"
    stress["variable_query_supported"] = True
    stress["native_only"] = False
    stress["source_schema"] = "figure5-validation-v4-scale-stress-1"
    stress["query_spec_hash"] = declared_spec_hash
    # The combined runner table is split into the two plotting tables while
    # retaining the exact same row keys and provenance columns.
    latency = stress.copy()
    memory = stress.copy()
    native_latency = v3_native["latency"].copy()
    native_memory = v3_native["memory"].copy()
    latency = pd.concat([native_latency, latency], ignore_index=True)
    memory = pd.concat([native_memory, memory], ignore_index=True)
    run.update(
        {
            "latency": latency,
            "memory": memory,
            "support": support,
            "boundary": canonical_boundary,
            "query_manifest": queries,
            "query_spec_hash": declared_spec_hash,
            "v3_native_sources": {
                "latency": str(v3_native["directory"] / "query_latency_summary.csv"),
                "memory": str(v3_native["directory"] / "memory_summary.csv"),
            },
        }
    )
    return run, []


def load_figure5_v4_data(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[SourceRecord]]:
    """Load V4 sources; no V3/V2 fallback is performed for panels d or e."""

    uq, uq_errors = _load_v3_uq(config, repo_root)
    native, native_errors = _load_v3_native(config, repo_root)
    training, training_errors = _load_training_cost(config, repo_root)
    scale, scale_errors = _load_scale_stress(config, repo_root)

    modes = {
        "a": "formal" if uq else "pending",
        "b": "formal" if uq else "pending",
        "c": "formal" if native else "pending",
        "d": "formal" if training else "pending",
        "e": "formal" if scale else "pending",
    }
    sources = {
        "a": str(_repo_path(repo_root, config["formal_inputs"]["uq_root"]) / str(config["formal_inputs"]["uq_run_id"]) / "crps_summary.csv"),
        "b": str(_repo_path(repo_root, config["formal_inputs"]["uq_root"]) / str(config["formal_inputs"]["uq_run_id"]) / "spread_error_summary.csv"),
        "c": str(_repo_path(repo_root, config["formal_inputs"]["cost_root"]) / str(config["formal_inputs"]["cost_run_id"]) / "native_summary.csv"),
        "d": str(_repo_path(repo_root, config["formal_inputs"]["training_cost_root"]) / str(config["formal_inputs"]["training_cost_run_id"]) / "training_cost_summary.csv"),
        "e": str(_repo_path(repo_root, config["formal_inputs"]["scale_root"]) / str(config["formal_inputs"]["scale_run_id"]) / "scale_stress_summary.csv"),
    }
    notes = {
        "a": "V3 formal paired normalized empirical CRPS reused unchanged." if uq else "Requires the V3 formal five-method UQ run.",
        "b": "V3 formal macro spread/error association reused unchanged; not calibration." if uq else "Requires the V3 formal cross-model UQ run.",
        "c": "V3 formal clean native model-core timing reused unchanged." if native else "Requires the V3 formal clean native benchmark.",
        "d": "V4 training-compute source; no V3 query-latency fallback is permitted." if training else "Requires a passing V4 training-cost manifest and summary table.",
        "e": "V4 Sobol high-N latency/memory stress source merged with validated V3 native rows; common query hashes and explicit throughput-only region are enforced." if scale else "Requires a passing V4 high-N stress bundle; V3 query/memory tables alone are not accepted.",
    }
    errors_by_panel = {
        "a": uq_errors,
        "b": uq_errors,
        "c": native_errors,
        "d": training_errors,
        "e": scale_errors,
    }
    records = []
    for panel in "abcde":
        errors = errors_by_panel[panel]
        note = notes[panel]
        if errors:
            note = f"{note} Diagnostic: {'; '.join(errors)}"
        records.append(SourceRecord(panel, modes[panel], "available" if modes[panel] == "formal" else "missing", sources[panel], note))

    return {
        "modes": modes,
        "sources": sources,
        "source_errors": errors_by_panel,
        "uq_crps": None if uq is None else uq["crps"],
        "uq_spread": None if uq is None else uq["spread"],
        "cost_native": None if native is None else native["native"],
        "training_cost": None if training is None else training["training"],
        "training_metric": None if training is None else training["metric_name"],
        "training_metric_label": None if training is None else training["metric_label"],
        "scale_latency": None if scale is None else scale["latency"],
        "scale_memory": None if scale is None else scale["memory"],
        "query_support": None if scale is None else scale["support"],
        "scale_boundary": None if scale is None else scale["boundary"],
        "query_manifest": None if scale is None else scale["query_manifest"],
        "run_metadata": {"uq": uq, "native": native, "training": training, "scale": scale},
    }, records

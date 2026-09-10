#!/usr/bin/env python
"""Build the additive Figure 5 V7R2 supplementary package.

The main V7R2 renderer owns the plotting language and the saved-table loading
contract.  This module only composes supplementary figures and records their
source coordinates/QA.  It intentionally has no raw-cache
or model-inference fallback: a timestamped ``source_qa.json`` with
``status == pass`` and the V7R2 renderer's five derived tables are required
before any formal artwork is written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[3]
ROOT_RENDERER_PATH = PACKAGE_ROOT / "figures" / "scripts" / "build_figure5_v7r2.py"

MAIN_METHODS = ["A0", "A2", "A3", "A5", "A4", "Senseiver"]
ABLATION_METHODS = ["A0", "A2", "A3", "A5", "A4", "A1"]
CONTROL_METHODS = ["A0", "A1", "Senseiver"]
FIELDS = ["CH4", "CO", "T", "U1", "p"]
FIELD_EXPORT = {"CH4": "Y_CH4", "CO": "Y_CO", "T": "T", "U1": "U1", "p": "p"}
FIELD_ALIASES = {
    "Y_CH4": "CH4",
    "CH4": "CH4",
    "Y_CO": "CO",
    "CO": "CO",
    "T": "T",
    "U1": "U1",
    "p": "p",
}
MACRO_FIELD = "Unobserved_mean"
POLICY_VALUES = {"last", "last.pt", "last_pt"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_root_renderer() -> Any:
    if not ROOT_RENDERER_PATH.is_file():
        raise FileNotFoundError(
            f"required V7R2 renderer is missing: {_relpath(ROOT_RENDERER_PATH)}; "
            "SI rendering cannot choose a separate source or style"
        )
    spec = importlib.util.spec_from_file_location("figure5_v7r2_renderer_for_si", ROOT_RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load V7R2 renderer: {ROOT_RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = ("configure_style", "load_tables", "draw_distribution", "draw_spectra", "save")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"V7R2 renderer is missing shared SI API: {', '.join(missing)}")
    return module


def _canonical_field(value: Any) -> str:
    text = str(value)
    return FIELD_ALIASES.get(text, text)


def _normalise_table(name: str, table: Any) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)
    frame = table.copy()
    if "method" not in frame.columns:
        raise ValueError(f"{name} is missing required method column")
    if "field" not in frame.columns:
        for candidate in ("target", "field_name", "variable"):
            if candidate in frame.columns:
                frame["field"] = frame[candidate]
                break
    if "field" not in frame.columns:
        raise ValueError(f"{name} is missing required field/target column")
    frame["field"] = frame["field"].map(_canonical_field)
    frame["method"] = frame["method"].astype(str)
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    for column in ("mean", "median", "q25", "q75", "p95", "max", "block20_ci95_low", "block20_ci95_high", "absoluteenergy", "absolute_energy"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _is_last(value: Any) -> bool:
    text = str(value).strip().lower().replace(" ", "")
    return text in POLICY_VALUES


def _policy_filter(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Keep the explicitly requested last checkpoint rows.

    Tables with both policies are filtered.  Tables without a policy column
    are accepted only because ``load_tables`` is the V7R2 contract and already
    returns the primary last.pt table.  No best.pt row is used as a fallback.
    """

    policy_columns = [column for column in ("policy", "checkpoint_policy", "checkpoint_name", "checkpoint") if column in frame.columns]
    if not policy_columns:
        return frame
    for column in policy_columns:
        values = {str(value).strip().lower().replace(" ", "") for value in frame[column].dropna().unique()}
        if not values:
            continue
        if values.issubset(POLICY_VALUES):
            return frame.loc[frame[column].map(_is_last)].copy()
        if "best" in values or "best.pt" in values or "best_pt" in values:
            last_mask = frame[column].map(_is_last)
            if not bool(last_mask.any()):
                raise ValueError(f"{name} contains only best.pt rows; last.pt is required for SI")
            return frame.loc[last_mask].copy()
    raise ValueError(f"{name} has an ambiguous checkpoint policy: {policy_columns}")


def _require_columns(frame: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _require_methods(frame: pd.DataFrame, name: str, methods: Sequence[str], field: str | None = None) -> None:
    selected = frame if field is None else frame.loc[frame.field.eq(field)]
    missing = [method for method in methods if method not in set(selected.method)]
    if missing:
        suffix = f" for field {field}" if field is not None else ""
        raise ValueError(f"{name} is missing required methods{suffix}: {', '.join(missing)}")


def _require_states(frame: pd.DataFrame, name: str, methods: Sequence[str], field: str) -> None:
    _require_columns(frame, name, ("method", "field", "snapshot", "time_index", "value"))
    selected = frame.loc[frame.field.eq(field)]
    for method in methods:
        rows = selected.loc[selected.method.eq(method)]
        if len(rows) != 1000:
            raise ValueError(f"{name} has {len(rows)} rows for {method}/{field}; expected exactly 1000")
        if rows[["snapshot", "time_index"]].duplicated().any():
            raise ValueError(f"{name} has duplicate state identities for {method}/{field}")
        if not np.isfinite(rows.value.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} has non-finite values for {method}/{field}")


def _require_summary(frame: pd.DataFrame, name: str, methods: Sequence[str], field: str) -> None:
    _require_columns(frame, name, ("method", "field", "n", "mean", "block20_ci95_low", "block20_ci95_high", "median", "q25", "q75", "p95", "max"))
    selected = frame.loc[frame.field.eq(field)]
    for method in methods:
        rows = selected.loc[selected.method.eq(method)]
        if len(rows) != 1:
            raise ValueError(f"{name} has {len(rows)} rows for {method}/{field}; expected exactly one")
        row = rows.iloc[0]
        if int(row.n) != 1000:
            raise ValueError(f"{name} has n={row.n} for {method}/{field}; expected 1000")
        vals = row[["mean", "block20_ci95_low", "block20_ci95_high", "median", "q25", "q75", "p95", "max"]].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError(f"{name} has non-finite summary values for {method}/{field}")
        if not float(row["block20_ci95_low"]) <= float(row["mean"]) <= float(row["block20_ci95_high"]):
            raise ValueError(f"{name} has invalid CI ordering for {method}/{field}")


def _source_record(path: Path, role: str, rows: int | None = None) -> dict[str, Any]:
    return {
        "path": _relpath(path),
        "role": role,
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
        "rows": int(rows) if rows is not None else None,
    }


def _blocker(timestamp: str, errors: Sequence[str], *, source_qa: Any = None) -> int:
    derived = PACKAGE_ROOT / "results" / "derived" / timestamp
    docs = PACKAGE_ROOT / "docs" / "generated" / timestamp
    payload = {
        "schema_version": "figure5-v7r2-si-blocker-1",
        "timestamp": timestamp,
        "status": "blocked",
        "source_qa": source_qa,
        "errors": list(errors),
        "policy": "last.pt required for all new SI figures and tables",
    }
    _json_write(derived / "si_qa.json", payload)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "si_source_blocker.md").write_text(
        "# Figure 5 V7R2 SI source blocker\n\n"
        f"Release `{timestamp}` is **blocked** before formal rendering.\n\n"
        "Required source checks were not satisfied; no alternative source or checkpoint was substituted.\n\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n",
        encoding="utf-8",
    )
    return 1


def _prepare_tables(timestamp: str, renderer: Any) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[str]]:
    derived = PACKAGE_ROOT / "results" / "derived" / timestamp
    source_qa_path = derived / "source_qa.json"
    if not source_qa_path.is_file():
        return {}, {}, [f"missing source gate: {_relpath(source_qa_path)}"]
    try:
        source_qa = json.loads(source_qa_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {}, [f"cannot parse source gate {_relpath(source_qa_path)}: {exc}"]
    if source_qa.get("status") != "pass":
        return {}, source_qa, [f"source gate {_relpath(source_qa_path)} has status {source_qa.get('status')!r}; formal SI rendering requires pass"]
    try:
        loaded = renderer.load_tables(timestamp)
    except Exception as exc:  # preserve a concise blocker for the release ledger
        return {}, source_qa, [f"V7R2 load_tables failed after source gate pass: {type(exc).__name__}: {exc}"]
    required = ("reconstruction_states", "reconstruction_summary", "highband_states", "highband_summary", "spectra_population")
    missing = [name for name in required if name not in loaded]
    if missing:
        return {}, source_qa, [f"V7R2 load_tables omitted required SI tables: {', '.join(missing)}"]
    tables = {name: _policy_filter(_normalise_table(name, loaded[name]), name) for name in required}
    return tables, source_qa, []


def _compact_records(records: Sequence[Mapping[str, Any]], *, field: str | None = None) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        state_rows = item.pop("state_source_rows", None)
        if state_rows is not None:
            item["state_source_row_count"] = len(state_rows)
            if state_rows:
                item["state_source_row_min"] = int(min(state_rows))
                item["state_source_row_max"] = int(max(state_rows))
        if field is not None:
            item.setdefault("field_export", FIELD_EXPORT.get(field, field))
        compact.append(item)
    return compact


def _tag(fig: Any, label: str) -> None:
    fig.text(0.035, 0.965, label, fontsize=8.5, fontweight="bold", va="top", ha="left")


def _display_label(renderer: Any, method: str) -> str:
    styles = getattr(renderer, "STYLES", {})
    style = styles.get(method, {}) if isinstance(styles, Mapping) else {}
    return str(style.get("label", method))


def _draw_fieldwise_control(ax: Any, summary: pd.DataFrame, renderer: Any, methods: Sequence[str], records: list[dict[str, Any]]) -> None:
    method_offsets = np.linspace(-0.22, 0.22, len(methods))
    y_base = np.arange(len(FIELDS))[::-1]
    for field_index, field in enumerate(FIELDS):
        y = float(y_base[field_index])
        for method, offset in zip(methods, method_offsets):
            row = summary.loc[summary.field.eq(field) & summary.method.eq(method)]
            if len(row) != 1:
                raise ValueError(f"ambiguous fieldwise control summary for {method}/{field}")
            row = row.iloc[0]
            mean = float(row["mean"])
            low = float(row["block20_ci95_low"])
            high = float(row["block20_ci95_high"])
            style = renderer.STYLES[method]
            ax.errorbar(mean, y + float(offset), xerr=[[mean - low], [high - mean]], fmt=style["marker"], color=style["color"],
                        mfc="white" if method in {"A0", "Senseiver"} else style["color"], mec=style["color"], ms=4.0, mew=.75,
                        lw=.65, capsize=1.5, zorder=4)
            records.append({"method": method, "field": field, "field_export": FIELD_EXPORT[field], "mean": mean,
                            "ci_low": low, "ci_high": high, "n": int(row["n"]), "y": y + float(offset)})
    ax.set_yticks(y_base, [FIELD_EXPORT[field] for field in FIELDS])
    ax.set_ylabel("Field")
    ax.set_xlabel(r"Physical relative $L_2$")
    ax.set_xlim(left=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6, width=.6, length=2)
    handles = [Line2D([], [], marker=renderer.STYLES[m]["marker"], color=renderer.STYLES[m]["color"],
                       markerfacecolor="white" if m in {"A0", "Senseiver"} else renderer.STYLES[m]["color"],
                       lw=0, ms=4, label=_display_label(renderer, m)) for m in methods]
    ax.legend(handles=handles, fontsize=5.3, frameon=False, loc="upper right", handletextpad=.25, labelspacing=.2)


def build(timestamp: str, *, strict_formal: bool = False) -> int:
    renderer: Any | None = None
    try:
        renderer = _load_root_renderer()
    except Exception as exc:
        return _blocker(timestamp, [f"cannot load V7R2 renderer: {type(exc).__name__}: {exc}"])
    tables, source_qa, errors = _prepare_tables(timestamp, renderer)
    if errors:
        return _blocker(timestamp, errors, source_qa=source_qa)

    derived_dir = PACKAGE_ROOT / "results" / "derived" / timestamp
    figure_dir = PACKAGE_ROOT / "figures" / "generated" / timestamp / "si"
    figure_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    recon_states = tables["reconstruction_states"]
    recon_summary = tables["reconstruction_summary"]
    high_states = tables["highband_states"]
    high_summary = tables["highband_summary"]
    spectra = tables["spectra_population"]

    validation_errors: list[str] = []
    try:
        _require_states(recon_states, "reconstruction_states", ABLATION_METHODS, MACRO_FIELD)
        _require_summary(recon_summary, "reconstruction_summary", ABLATION_METHODS, MACRO_FIELD)
        _require_methods(high_summary, "highband_summary", MAIN_METHODS, "U1")
        _require_methods(high_states, "highband_states", MAIN_METHODS, "U1")
        _require_summary(high_summary, "highband_summary", MAIN_METHODS, "U1")
        _require_states(high_states, "highband_states", MAIN_METHODS, "U1")
        for field in FIELDS:
            _require_methods(spectra, "spectra_population", ["Truth", *MAIN_METHODS], field)
            _require_summary(high_summary, "highband_summary", MAIN_METHODS, field)
            _require_states(high_states, "highband_states", MAIN_METHODS, field)
        _require_states(recon_states, "reconstruction_states", CONTROL_METHODS, MACRO_FIELD)
        _require_summary(recon_summary, "reconstruction_summary", CONTROL_METHODS, MACRO_FIELD)
        for field in FIELDS:
            _require_summary(recon_summary, "reconstruction_summary", CONTROL_METHODS, field)
    except Exception as exc:
        validation_errors.append(f"SI table validation failed: {type(exc).__name__}: {exc}")

    provenance_path = derived_dir / "checkpoint_provenance.csv"
    if not provenance_path.is_file():
        validation_errors.append(f"missing required checkpoint provenance: {_relpath(provenance_path)}")
    else:
        try:
            provenance = _policy_filter(pd.read_csv(provenance_path), "checkpoint_provenance")
            if "method" not in provenance.columns:
                validation_errors.append("checkpoint_provenance.csv is missing method")
            else:
                missing = [method for method in ABLATION_METHODS if method not in set(provenance.method.astype(str))]
                if missing:
                    validation_errors.append("checkpoint_provenance.csv is missing ablation methods: " + ", ".join(missing))
        except Exception as exc:
            validation_errors.append(f"cannot read checkpoint_provenance.csv: {type(exc).__name__}: {exc}")

    if validation_errors:
        return _blocker(timestamp, validation_errors, source_qa=source_qa)

    renderer.configure_style()
    records: dict[str, list[dict[str, Any]]] = {"S1": [], "S2": [], "S3": []}
    manifest: dict[str, Any] = {
        "schema_version": "figure5-v7r2-si-manifest-1",
        "timestamp": timestamp,
        "status": "pending",
        "renderer": _relpath(ROOT_RENDERER_PATH),
        "policy": "last.pt",
        "source_gate": source_qa,
        "sources": {
            "reconstruction_states": _source_record(derived_dir / "reconstruction_states.csv", "V7R2 saved derived state table"),
            "reconstruction_summary": _source_record(derived_dir / "reconstruction_summary.csv", "V7R2 saved derived summary table"),
            "highband_states": _source_record(derived_dir / "highband_states.csv", "V7R2 saved derived high-band state table"),
            "highband_summary": _source_record(derived_dir / "highband_summary.csv", "V7R2 saved derived high-band summary table"),
            "spectra_population": _source_record(derived_dir / "spectra_population.csv", "V7R2 saved derived population spectrum table"),
            "checkpoint_provenance": _source_record(provenance_path, "saved checkpoint provenance"),
            "source_qa": _source_record(derived_dir / "source_qa.json", "mandatory source gate"),
        },
        "figures": [],
        "plot_coordinates": records,
        "caveats": [
            "All new SI figures use last.pt rows only; SI tables/captions are generated by the documentation stage.",
            "A1 is retained in SI as a deterministic-objective control and is excluded from the main ablation panels.",
            "Saved ablation checkpoints have unequal final epochs; this is a checkpoint-based comparison, not equal-budget retraining.",
        ],
    }

    def save_figure(fig: Any, name: str, group: str) -> None:
        path = figure_dir / f"{name}_{timestamp}.svg"
        entry = renderer.save(fig, path)
        manifest["figures"].append(entry)
        entry["group"] = group

    # SI Figure S1: six ablations including A1, with all statewise values.
    fig, ax = plt.subplots(figsize=(140 / 25.4, 62 / 25.4))
    s1_records = renderer.draw_distribution(ax, recon_states, recon_summary, ABLATION_METHODS, MACRO_FIELD,
                                            r"Unobserved-field relative $L_2$", fontsize=6, record=None, wrap_labels=True)
    records["S1"].extend(_compact_records(s1_records, field=MACRO_FIELD))
    _tag(fig, "S1")
    fig.subplots_adjust(left=.30, right=.98, bottom=.22, top=.92)
    save_figure(fig, "si_figure_s1_ablation_error_distributions", "S1")

    # SI Figure S2: one complete panel-e counterpart per candidate field and
    # an all-field entry figure.  The same root methods own both sub-axes.
    fig_s2, axes_s2 = plt.subplots(len(FIELDS), 2, figsize=(183 / 25.4, 250 / 25.4), squeeze=False)
    for index, field in enumerate(FIELDS):
        ax_top, ax_bottom = axes_s2[index]
        spectrum_records = renderer.draw_spectra(ax_top, spectra, field, methods=MAIN_METHODS, fontsize=5.6, legend=False, record=None)
        high_records = renderer.draw_distribution(ax_bottom, high_states, high_summary, MAIN_METHODS, field,
                                                   r"High-band relative $L_2$", fontsize=5.4, record=None, wrap_labels=True)
        records["S2"].extend(_compact_records(spectrum_records, field=field))
        records["S2"].extend(_compact_records(high_records, field=field))
        ax_top.text(.015, .94, FIELD_EXPORT[field], transform=ax_top.transAxes, fontsize=6, fontweight="bold", va="top")
    _tag(fig_s2, "S2")
    # The log-scale tick labels can be wider than the field labels (especially
    # for the temperature and pressure spectra).  Keep a real left/bottom
    # margin so the editable SVG has no text outside the canvas.
    fig_s2.subplots_adjust(left=.27, right=.985, bottom=.06, top=.98, hspace=.88, wspace=.35)
    save_figure(fig_s2, "si_figure_s2_scale_resolved_all_fields", "S2")

    # Standalone complete panel-e alternatives, preserving the candidate list
    # in filenames and manifest records for author selection.
    for field in FIELDS:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(89 / 25.4, 112 / 25.4))
        spectrum_records = renderer.draw_spectra(ax_top, spectra, field, methods=MAIN_METHODS, fontsize=5.8, legend=False, record=None)
        high_records = renderer.draw_distribution(ax_bottom, high_states, high_summary, MAIN_METHODS, field,
                                                   r"High-band relative $L_2$", fontsize=5.7, record=None, wrap_labels=True)
        ax_top.text(.02, .95, FIELD_EXPORT[field], transform=ax_top.transAxes, fontsize=6.5, fontweight="bold", va="top")
        _tag(fig, "e")
        fig.subplots_adjust(left=.45, right=.98, bottom=.18, top=.95, hspace=.58)
        export = FIELD_EXPORT[field]
        save_figure(fig, f"panel_e_{export}", "S2_panel_e_alternative")
        records["S2"].extend(_compact_records(spectrum_records, field=field))
        records["S2"].extend(_compact_records(high_records, field=field))

    # SI Figure S3: macro distribution and fieldwise physical relative-L2
    # control.  The right axis uses source summaries and intervals directly.
    fig, (ax_macro, ax_fields) = plt.subplots(1, 2, figsize=(183 / 25.4, 72 / 25.4), gridspec_kw={"width_ratios": [1.0, 1.25]})
    macro_records = renderer.draw_distribution(ax_macro, recon_states, recon_summary, CONTROL_METHODS, MACRO_FIELD,
                                               r"Unobserved-field relative $L_2$", fontsize=5.8, record=None, wrap_labels=True)
    records["S3"].extend(_compact_records(macro_records, field=MACRO_FIELD))
    _draw_fieldwise_control(ax_fields, recon_summary, renderer, CONTROL_METHODS, records["S3"])
    _tag(fig, "S3")
    fig.subplots_adjust(left=.16, right=.99, bottom=.22, top=.92, wspace=.36)
    save_figure(fig, "si_figure_s3_deterministic_objective_control", "S3")

    # Machine-readable source coordinates omit the large raw row-index lists
    # returned by the root distribution helper but retain all plotted summary
    # values, shell coordinates, methods, fields and checkpoint policy.
    _json_write(derived_dir / "si_source_coordinates.json", {"schema_version": "figure5-v7r2-si-coordinates-1", "timestamp": timestamp, "policy": "last.pt", "coordinates": records})
    manifest["qa"] = {
        "status": "pass" if all(not item.get("off_canvas_text") for item in manifest["figures"]) else "layout_review_required",
        "off_canvas_text": [
            {"path": item["path"], "text": item.get("off_canvas_text", [])}
            for item in manifest["figures"]
            if item.get("off_canvas_text")
        ],
        "png_dpi": 600,
    }
    manifest["status"] = manifest["qa"]["status"]
    _json_write(derived_dir / "si_plot_manifest.json", manifest)
    _json_write(figure_dir / "si_plot_manifest.json", manifest)
    _json_write(derived_dir / "si_qa.json", {
        "schema_version": "figure5-v7r2-si-qa-1",
        "timestamp": timestamp,
        "status": manifest["status"],
        "source_gate_status": source_qa.get("status"),
        "checkpoint_policy": "last.pt",
        "figures": [{"path": item["path"], "off_canvas_text": item.get("off_canvas_text", [])} for item in manifest["figures"]],
        "checks": {
            "s1_includes_a1": "A1" in set(recon_summary.loc[recon_summary.field.eq(MACRO_FIELD), "method"]),
            "s1_has_statewise_distributions": all(len(recon_states.loc[recon_states.field.eq(MACRO_FIELD) & recon_states.method.eq(method)]) == 1000 for method in ABLATION_METHODS),
            "s2_fields": [FIELD_EXPORT[field] for field in FIELDS],
            "s2_includes_truth_and_senseiver": all({"Truth", "Senseiver"}.issubset(set(spectra.loc[spectra.field.eq(field), "method"])) for field in FIELDS),
            "s3_includes_deterministic_control": set(CONTROL_METHODS).issubset(set(recon_summary.loc[recon_summary.field.eq(MACRO_FIELD), "method"])),
            "last_only": True,
            "unequal_epochs_caveat_visible": True,
            "all_svg_off_canvas_checks_pass": manifest["qa"]["status"] == "pass",
        },
    })
    print(json.dumps({"status": manifest["status"], "timestamp": timestamp, "figures": len(manifest["figures"])}))
    if strict_formal and manifest["status"] != "pass":
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--strict-formal", action="store_true")
    args = parser.parse_args(argv)
    status = build(args.timestamp, strict_formal=args.strict_formal)
    if args.strict_formal and status != 0:
        raise SystemExit(status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

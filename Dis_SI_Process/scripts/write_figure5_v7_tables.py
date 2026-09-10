#!/usr/bin/env python
"""Write the compact Figure 5 V7 LaTeX table package.

The table writer consumes the timestamped, source-validated CSV reductions
from ``collect_figure5_v7_ablation_sources.py``.  It deliberately does not
recreate values from the prose evaluation report: means, intervals, paired
effects, fractions, provenance and rare-event identities are read from the
CSV rows and rendered into short policy/metric tables.  The output is placed
under ``docs/generated/<timestamp>/latex/tables`` so the parent document
writer can include the group index ``latex/si_ablation_tables.tex``.

This stage is saved-data-only.  It never loads a checkpoint, reads a
reconstruction cache, launches inference, or writes source data.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT.parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

STOCHASTIC_ORDER = ("A0", "A2", "A3", "A5", "A4")
ALL_METHOD_ORDER = (*STOCHASTIC_ORDER,"A1")
UNOBSERVED_TARGETS = ("CH4", "CO", "U1", "p")
FIELD_ORDER = ("CH4", "CO", "T", "U1", "p")
PAIR_ORDER = ("T-U1", "CH4-U1", "p-U1")

LABELS = {
    "A0": "Full model",
    "A1": "Deterministic regression",
    "A2": "No sensor feedback",
    "A3": "No local conditioning",
    "A4": "IID Gaussian prior",
    "A5": "Local-only conditioning",
}

RAW_TEX_PREFIX = "__FIGURE5_RAW_TEX__"


def _read(path: Path, *, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Add tolerant aliases without changing source columns."""

    if df.empty:
        return df.copy()
    out = df.copy()
    if "policy" not in out and "checkpoint_policy" in out:
        out["policy"] = out["checkpoint_policy"]
    if "field" not in out and "target" in out:
        out["field"] = out["target"]
    if "target" not in out and "field" in out:
        out["target"] = out["field"]
    if "display_label" not in out and "method" in out:
        out["display_label"] = out["method"].map(LABELS).fillna(out["method"])
    # A few early collector revisions called the interval columns low/high.
    aliases = {
        "method_mean_minus_baseline": "mean_difference",
        "mean_ci_low": "block20_ci95_low",
        "mean_ci_high": "block20_ci95_high",
        "ci95_low": "block20_ci95_low",
        "ci95_high": "block20_ci95_high",
        "low": "block20_ci95_low",
        "high": "block20_ci95_high",
    }
    for source, destination in aliases.items():
        if destination not in out and source in out:
            out[destination] = out[source]
    return out


def _numeric(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, *, digits: int = 8, percent: bool = False, integer: bool = False) -> str:
    x = _numeric(value)
    if x is None:
        return "--"
    if integer:
        return f"{int(round(x)):,}"
    if percent:
        # Return plain text here.  Data cells are escaped exactly once by
        # ``_escape``; returning a TeX command would turn ``\%`` into a
        # literal ``\textbackslash`` sequence in the rendered table.
        return f"{100.0*x:.{max(1, digits - 2)}f}%"
    # Keep enough digits to expose source-level differences while avoiding a
    # forest of insignificant binary digits in the generated tables.
    return f"{x:.{digits}g}"


def _fmt_change(value: Any) -> str:
    x = _numeric(value)
    if x is None:
        return "--"
    return f"{x:+.5f}"


def _fmt_pct_change(value: Any) -> str:
    x = _numeric(value)
    if x is None:
        return "--"
    return f"{x:+.2f}%"


def _raw_tex(value: str) -> str:
    """Mark a deliberately authored TeX cell for ``_escape``."""

    return RAW_TEX_PREFIX + value


def _breakable_token(value: Any, *, chunk: int = 8) -> str:
    """Return an exact token with zero-width TeX break opportunities."""

    text = "" if value is None else str(value)
    if not text or text in {"nan", "None", "NaN"}:
        return "--"
    # Hashes and paths are kept exactly, but an unbreakable 64-character hash
    # or checkpoint path can otherwise force a table past the text block.
    escaped = text.replace("\\", r"\textbackslash{}").replace("%", r"\%")
    escaped = escaped.replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")
    escaped = escaped.replace("{", r"\{").replace("}", r"\}")
    if "/" in escaped:
        escaped = escaped.replace("/", r"/\allowbreak{}").replace(r"\_",r"\_\allowbreak{}")
    elif chunk > 0:
        escaped = r"\allowbreak{}".join(escaped[i : i + chunk] for i in range(0, len(escaped), chunk))
    return _raw_tex(r"\texttt{" + escaped + "}")


def _fmt_interval(row: Mapping[str, Any], *, mean_key: str = "mean", low_key: str = "block20_ci95_low", high_key: str = "block20_ci95_high", digits: int = 8) -> str:
    mean = _fmt(row.get(mean_key), digits=digits)
    low = _fmt(row.get(low_key), digits=digits)
    high = _fmt(row.get(high_key), digits=digits)
    if mean == "--":
        return "--"
    return f"{mean} [{low}, {high}]"


def _fmt_effect(row: Mapping[str, Any], *, digits: int = 8) -> str:
    delta = _fmt(row.get("mean_difference"), digits=digits)
    low = _fmt(row.get("block20_ci95_low"), digits=digits)
    high = _fmt(row.get("block20_ci95_high"), digits=digits)
    if delta == "--":
        return "--"
    return f"{delta} [{low}, {high}]"


def _escape(value: Any) -> str:
    """Escape ordinary text for a LaTeX table cell."""

    text = "" if value is None else str(value)
    if text in {"", "nan", "None", "NaN"}:
        return "--"
    if text.startswith(RAW_TEX_PREFIX):
        return text[len(RAW_TEX_PREFIX) :]
    text = text.replace("\\", r"\textbackslash{}")
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _display_method(value: Any) -> str:
    method = str(value)
    return LABELS.get(method, method)


def _row(table: pd.DataFrame, **filters: Any) -> pd.Series | None:
    if table.empty:
        return None
    rows = table
    for key, value in filters.items():
        if key not in rows:
            return None
        rows = rows.loc[rows[key].astype(str).eq(str(value))]
    if rows.empty:
        return None
    return rows.iloc[0]


def _sort_methods(frame: pd.DataFrame, *, methods: Sequence[str] = ALL_METHOD_ORDER) -> pd.DataFrame:
    if frame.empty or "method" not in frame:
        return frame
    order = {m: i for i, m in enumerate(methods)}
    out = frame.copy()
    out["__method_order"] = out["method"].astype(str).map(order).fillna(999)
    sort_cols = ["__method_order"]
    if "policy" in out:
        out["__policy_order"] = out["policy"].astype(str).map({"last": 0, "best": 1}).fillna(999)
        sort_cols.append("__policy_order")
    out = out.sort_values(sort_cols, kind="stable")
    return out.drop(columns=[c for c in ("__method_order", "__policy_order") if c in out])


def _cell(row: pd.Series | None, *, digits: int = 8) -> str:
    return "--" if row is None else _fmt_interval(row, digits=digits)


def _effect_cell(row: pd.Series | None, *, digits: int = 8) -> str:
    return "--" if row is None else _fmt_effect(row, digits=digits)


def _write_table(path: Path, *, caption: str, label: str, columns: Sequence[str], rows: Sequence[Sequence[Any]], alignment: str | None = None, note: str | None = None) -> None:
    """Split wide column groups and paginate at an ordinary nine-point size.

    Explicit paragraph widths preserve the 183-mm manuscript text block.
    Longtable repeats headings across pages; no text is scaled or discarded.
    """
    for index, row in enumerate(rows):
        if len(row) != len(columns):
            raise ValueError(f"{path.name}: row {index} has {len(row)} cells for {len(columns)} headers")
    path.parent.mkdir(parents=True, exist_ok=True)
    identity_names={'Configuration','Policy','Target','Field','Pair','Metric'}
    identity=[]
    for i,c in enumerate(columns):
        if c in identity_names:identity.append(i)
        else:break
    values=list(range(len(identity),len(columns)))
    group_size=max(1,5-len(identity))
    groups=[identity+values[i:i+group_size] for i in range(0,len(values),group_size)] or [list(range(len(columns)))]
    lines=['% Generated from validated compact source rows; no resizebox or reduced-size text.']
    for g,indices in enumerate(groups):
        weights=[]
        for i in indices:
            col=columns[i]
            if col=='Configuration':w=40
            elif col=='Policy':w=12
            elif col=='Target':w=25
            elif col=='Field':w=12
            elif col=='Pair':w=18
            elif col=='Metric':w=58
            elif 'SHA' in col or 'path' in col.lower() or 'directory' in col.lower():w=95
            elif 'objective' in col.lower():w=55
            elif any('[' in str(row[i]) for row in rows if len(row)>i):w=47
            else:w=27
            weights.append(w)
        available=182.6-2*(len(indices)-1)*3*25.4/72.27
        widths=[available*w/sum(weights) for w in weights]
        align='@{}'+''.join('>{\\raggedright\\arraybackslash}p{'+f'{w:.4f}'+'mm}' for w in widths)+'@{}'
        suffix='' if len(groups)==1 else f' Column group {g+1} of {len(groups)}.'
        tag=label if g==0 else label+f'-columns-{g+1}'
        heading=' & '.join(str(columns[i]) for i in indices)+r' \\'
        lines += [r'\begingroup',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.12}',
                  '\\begin{longtable}{'+align+'}',f'\\caption{{{caption}{suffix}}}\\label{{{tag}}}'+r' \\',
                  r'\toprule',heading,r'\midrule',r'\endfirsthead',r'\toprule',heading,r'\midrule',r'\endhead',
                  r'\bottomrule',r'\endfoot']
        for row in rows:
            cells=[]
            for i in indices:
                raw=row[i] if i<len(row) else '--'
                if isinstance(raw,(float,np.floating)):raw=_fmt(raw,digits=8)
                cell=_escape(raw)
                if not str(row[i] if i<len(row) else '').startswith(RAW_TEX_PREFIX):
                    cell=cell.replace(r'\_',r'\_\allowbreak{}').replace('/',r'/\allowbreak{}')
                cells.append(cell)
            lines.append(' & '.join(cells)+r' \\')
        lines += [r'\end{longtable}',r'\endgroup']
    # Notes, like captions, are authored TeX fragments rather than data cells.
    if note:lines += [r'\noindent '+note,r'\par\medskip']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def _write_input_index(path: Path, *, title: str, label: str, includes: Sequence[str], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated table-group index. Child tables are intentionally split by policy/metric for readable manuscript layouts.",
        f"\\begin{{table}}[htbp]",
        "\\centering",
        f"\\caption{{{_escape(title)}}}",
        f"\\label{{{_escape(label)}}}",
        "\\begin{minipage}{0.98\\linewidth}",
        f"\\small {_escape(note)}",
        "\\end{minipage}",
        "\\end{table}",
    ]
    # LaTeX resolves ``\input`` paths relative to the top-level document,
    # not relative to this child table.  Use the stable repository-relative
    # path so the generated package compiles from the repository root, which
    # is the bundle runner's copy-ready invocation mode.
    prefix = path.parent.relative_to(REPO_ROOT).as_posix()
    for item in includes:
        lines.append(f"\\input{{{prefix}/{item}}}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_value(summary: pd.DataFrame, policy: str, method: str, metric: str, target: str) -> pd.Series | None:
    return _row(summary, policy=policy, method=method, metric=metric, target=target)


def _write_mean_ci_metric(
    path: Path,
    *,
    frame: pd.DataFrame,
    metric: str,
    methods: Sequence[str],
    dimension_key: str,
    dimensions: Sequence[str],
    dimension_label: str,
    caption: str,
    label: str,
    note: str,
) -> None:
    """Write one compact source metric with separate mean/CI columns."""

    rows: list[list[Any]] = []
    for method in methods:
        for dimension in dimensions:
            filters = {"method": method, "metric": metric, dimension_key: dimension}
            row = _row(frame, **filters)
            if row is None:
                continue
            rows.append([
                _display_method(method), dimension,
                _fmt(row.get("mean"), digits=8),
                _fmt(row.get("block20_ci95_low"), digits=8),
                _fmt(row.get("block20_ci95_high"), digits=8),
            ])
    _write_table(
        path,
        caption=caption,
        label=label,
        columns=["Configuration", dimension_label, "Mean", "CI low", "CI high"],
        rows=rows,
        alignment="llrrr",
        note=note,
    )


def _write_effect_metric(
    path: Path,
    *,
    frame: pd.DataFrame,
    metric: str,
    methods: Sequence[str],
    dimension_key: str,
    dimensions: Sequence[str],
    dimension_label: str,
    caption: str,
    label: str,
    note: str,
    include_policy: bool = False,
) -> None:
    """Write one paired metric with numeric interval endpoints."""

    rows: list[list[Any]] = []
    for method in methods:
        for dimension in dimensions:
            row = _row(frame, method=method, baseline="A0", metric=metric, **{dimension_key: dimension})
            if row is None:
                continue
            prefix: list[Any] = [row.get("policy", "--")] if include_policy else []
            rows.append(prefix + [
                _display_method(method), dimension,
                _fmt(row.get("mean_difference"), digits=8),
                _fmt(row.get("block20_ci95_low"), digits=8),
                _fmt(row.get("block20_ci95_high"), digits=8),
                _fmt_pct_change(row.get("relative_mean_change_percent")),
                _fmt(row.get("fraction_method_less_than_baseline"), percent=True),
            ])
    columns = (["Policy"] if include_policy else []) + ["Configuration", dimension_label, "$\\Delta$ mean", "CI low", "CI high", "Relative change", "Fraction lower"]
    alignment = ("l" if include_policy else "") + "llrrrrr"
    _write_table(path, caption=caption, label=label, columns=columns, rows=rows, alignment=alignment, note=note)


def _provenance_table(prov: pd.DataFrame, out: Path) -> list[Path]:
    children: list[Path] = []
    rows: list[list[Any]] = []
    for method in ALL_METHOD_ORDER:
        frame = prov.loc[prov.method.astype(str).eq(method)] if not prov.empty and "method" in prov else pd.DataFrame()
        for policy in ("last", "best"):
            row = _row(frame, policy=policy)
            if row is None:
                continue
            rows.append([
                _display_method(method), policy,
                row.get("checkpoint_epoch", "--"), row.get("global_step", "--"),
                row.get("optimizer_state_parameter_elements", row.get("parameter_elements_with_optimizer_state", "--")),
                row.get("configured_epochs", "--"), row.get("learning_rate", "--"),row.get('scheduler_t_max','--'),
            ])
    path = out / "tab_ablation_provenance_checkpoint_ledger.tex"
    _write_table(
        path,
        caption="Ablation checkpoint ledger and training-budget metadata.",
        label="tab:ablation-provenance-ledger",
        columns=["Configuration", "Policy", "Epoch", "Global step", "Optimizer-state elements", "Configured epochs", "Learning rate","LR scheduler horizon"],
        rows=rows,
        alignment="llrrrrr",
        note="Optimizer-state elements are the recorded parameter elements with optimizer state. Epoch endpoints differ across saved runs; no epoch normalization is applied.",
    )
    children.append(path)

    # Hashes are kept in a companion so the ledger remains a compact numeric
    # table.  Zero-width break opportunities preserve the exact hash while
    # preventing an unbreakable token from overflowing the text block.
    rows = []
    for method in ALL_METHOD_ORDER:
        frame = prov.loc[prov.method.astype(str).eq(method)] if not prov.empty and "method" in prov else pd.DataFrame()
        for policy in ("last", "best"):
            row = _row(frame, policy=policy)
            if row is None:
                continue
            rows.append([
                _display_method(method), policy,
                _breakable_token(row.get("checkpoint_sha256", "--")),
            ])
    path = out / "tab_ablation_provenance_hashes.tex"
    _write_table(
        path,
        caption="Exact checkpoint SHA-256 identities.",
        label="tab:ablation-provenance-hashes",
        columns=["Configuration", "Policy", "Checkpoint SHA-256"],
        rows=rows,
        alignment="llp{0.62\\linewidth}",
        note="Hashes are copied from the accepted local checkpoint manifest; line breaks are zero-width TeX opportunities only.",
    )
    children.append(path)

    rows = []
    for method in ALL_METHOD_ORDER:
        frame = prov.loc[prov.method.astype(str).eq(method)] if not prov.empty and "method" in prov else pd.DataFrame()
        for policy in ("last", "best"):
            row = _row(frame, policy=policy)
            if row is None:
                continue
            rows.append([
                _display_method(method), policy,
                row.get("training_batch_size", "--"), row.get("training_query_count", "--"),
                row.get("selection_objective", "--"), row.get("train_loss", "--"), row.get("val_loss", "--"),
                _breakable_token(row.get("run_directory", "--")),
                _breakable_token(row.get("checkpoint_path", "--")),
            ])
    path = out / "tab_ablation_provenance_training_metadata.tex"
    _write_table(path, caption="Ablation training metadata, selection objective and endpoint caveats.", label="tab:ablation-provenance-training", columns=["Configuration", "Policy", "Train batch", "Train query count", "Selection objective", "Train loss", "Validation loss", "Run directory", "Checkpoint path"], rows=rows, alignment="llrrllrp{0.19\\linewidth}p{0.21\\linewidth}", note="Paths are retained with zero-width break opportunities; the source manifest retains the resolved absolute identities. A saved checkpoint comparison has matched states and measurements but unmatched training budgets.")
    children.append(path)

    master = out / "tab_ablation_provenance.tex"
    _write_input_index(master, title="Ablation provenance and checkpoint scope.", label="tab:ablation-provenance", includes=[p.name for p in children], note="The ledger preserves scientific names, both checkpoint policies, exact hashes, optimizer-state element counts and recorded training metadata.")
    children.insert(0, master)
    return children


def _reconstruction_tables(summary: pd.DataFrame, paired: pd.DataFrame, out: Path) -> list[Path]:
    children: list[Path] = []
    metrics = [
        ("physical_relative_l2", "Physical relative L2", "physical"),
        ("normalized_relative_l2", "Standardized relative L2", "standardized"),
        ("truth_fluctuation_normalized_l2", "Truth-fluctuation relative L2", "fluctuation"),
        ("physical_relative_l2_excluding_sensors", "Physical relative L2, sensor-excluded", "sensor_excluded"),
    ]
    targets = (*FIELD_ORDER, "Unobserved_mean", "All_fields_mean")

    # Keep each metric in its own five-column table.  A cell containing a
    # mean plus two interval endpoints is useful for reading but becomes
    # unacceptably wide when five such cells share one row.  The per-metric
    # split preserves the complete source values while fitting the 183-mm
    # manuscript text block at ordinary SI table size.
    for policy in ("last", "best"):
        policy_children: list[Path] = []
        for metric, metric_label, metric_slug in metrics:
            rows = []
            for method in STOCHASTIC_ORDER:
                for target in targets:
                    row = _summary_value(summary, policy, method, metric, target)
                    if row is None:
                        continue
                    rows.append([
                        _display_method(method), target,
                        _fmt(row.get("mean"), digits=8),
                        _fmt(row.get("block20_ci95_low"), digits=8),
                        _fmt(row.get("block20_ci95_high"), digits=8),
                    ])
            path = out / f"tab_ablation_reconstruction_{policy}_{metric_slug}.tex"
            _write_table(
                path,
                caption=f"{metric_label} by field and macro target ({policy}.pt).",
                label=f"tab:ablation-reconstruction-{policy}-{metric_slug}",
                columns=["Configuration", "Target", "Mean", "CI low", "CI high"],
                rows=rows,
                alignment="llrrr",
                note="Intervals are 95\% circular moving-block CIs with block length 20. The primary macro is the equal-weight mean of CH4, CO, U1 and p statewise relative-$L_2$ values; temperature is observed and excluded from that macro.",
            )
            policy_children.append(path)
            children.append(path)
        policy_index = out / f"tab_ablation_reconstruction_{policy}.tex"
        _write_input_index(
            policy_index,
            title=f"Reconstruction errors ({policy}.pt), split by metric.",
            label=f"tab:ablation-reconstruction-{policy}",
            includes=[p.name for p in policy_children],
            note="Each metric is rendered as a compact field/macro table so full-precision means and interval endpoints remain readable.",
        )
        children.append(policy_index)

    # Paired effects are also split by policy and metric.  Separate numeric
    # mean/low/high columns avoid an unbreakable ``mean [low, high]`` cell and
    # make the source CI endpoints inspectable in the generated artifact.
    paired_children: list[Path] = []
    if not paired.empty:
        for policy in ("last", "best"):
            frame = paired.loc[
                (paired.policy.astype(str) == policy)
                & paired.metric.astype(str).isin({metric for metric, _, _ in metrics[:3]})
            ]
            for metric, metric_label, metric_slug in metrics[:3]:
                rows = []
                for method in ("A2", "A3", "A5", "A4"):
                    for target in (*FIELD_ORDER, "Unobserved_mean"):
                        row = _row(frame, method=method, baseline="A0", metric=metric, target=target)
                        if row is None:
                            continue
                        rows.append([
                            _display_method(method), target,
                            _fmt(row.get("mean_difference"), digits=8),
                            _fmt(row.get("block20_ci95_low"), digits=8),
                            _fmt(row.get("block20_ci95_high"), digits=8),
                            _fmt_pct_change(row.get("relative_mean_change_percent")),
                            _fmt(row.get("fraction_method_less_than_baseline"), percent=True),
                        ])
                path = out / f"tab_ablation_reconstruction_paired_{policy}_{metric_slug}.tex"
                _write_table(
                    path,
                    caption=f"Paired {metric_label.lower()} effects versus the full reference ({policy}.pt).",
                    label=f"tab:ablation-reconstruction-paired-{policy}-{metric_slug}",
                    columns=["Configuration", "Target", "$\\Delta$ mean", "CI low", "CI high", "Relative change", "Fraction lower"],
                    rows=rows,
                    alignment="llrrrrr",
                    note="Paired differences are variant minus full, computed on exactly matched state/time identities. Fractions are state fractions, not training-repeat success rates.",
                )
                paired_children.append(path)
                children.append(path)
    paired_index = out / "tab_ablation_reconstruction_paired_effects.tex"
    _write_input_index(
        paired_index,
        title="Paired reconstruction effects versus the full reference.",
        label="tab:ablation-reconstruction-paired",
        includes=[p.name for p in paired_children],
        note="Paired effects retain separate policy/metric tables with full interval endpoints, relative mean changes and fractions improved.",
    )
    children.append(paired_index)

    contrast=paired.loc[paired.method.eq('A5')&paired.baseline.eq('A2')&paired.metric.eq('physical_relative_l2')]
    rows=[[r.policy,r.target,_fmt(r.mean_difference),_fmt(r.block20_ci95_low),_fmt(r.block20_ci95_high),_fmt_pct_change(r.relative_mean_change_percent),_fmt(r.fraction_method_less_than_baseline,percent=True)] for r in contrast.itertuples()]
    contrast_path=out/'tab_ablation_reconstruction_local_only_vs_no_feedback.tex'
    _write_table(contrast_path,caption='Local-only conditioning minus no sensor feedback: physical reconstruction contrast.',label='tab:ablation-local-only-vs-no-feedback',columns=['Policy','Target','Mean difference','CI low','CI high','Relative change','Fraction lower'],rows=rows,note='Both configurations have unequal effective optimized capacity. Positive differences denote larger local-only error; intervals use paired circular blocks of 20 states.')
    children.append(contrast_path)

    master = out / "tab_ablation_reconstruction.tex"
    _write_input_index(
        master,
        title="Ablation reconstruction errors and paired effects.",
        label="tab:ablation-reconstruction",
        includes=[
            "tab_ablation_reconstruction_last.tex",
            "tab_ablation_reconstruction_best.tex",
            "tab_ablation_reconstruction_paired_effects.tex",
            "tab_ablation_reconstruction_local_only_vs_no_feedback.tex",
        ],
        note="Per-field physical, standardized and fluctuation-normalized errors are separated from sensor-excluded checks; both policies and paired effects are retained.",
    )
    children.insert(0, master)
    return children


def _spectral_tables(spectral: pd.DataFrame, out: Path) -> list[Path]:
    children: list[Path] = []
    metrics = [
        ("spectral_low_energy_ratio", "Low-band energy ratio", "low"),
        ("spectral_mid_energy_ratio", "Middle-band energy ratio", "middle"),
        ("spectral_high_energy_ratio", "High-band energy ratio", "high"),
        ("spectral_total_energy_ratio", "Total energy ratio", "total"),
        ("spectral_lsd_db", "Whole-spectrum LSD (dB)", "lsd"),
    ]
    for policy in ("last", "best"):
        policy_children: list[Path] = []
        frame = spectral.loc[spectral.policy.astype(str).eq(policy)] if not spectral.empty and "policy" in spectral else pd.DataFrame()
        for metric, metric_label, metric_slug in metrics:
            rows = []
            for method in STOCHASTIC_ORDER:
                for field in FIELD_ORDER:
                    r = _row(frame, method=method, metric=metric, target=field)
                    if r is None:
                        continue
                    rows.append([
                        _display_method(method), field,
                        _fmt(r.get("mean"), digits=8),
                        _fmt(r.get("block20_ci95_low"), digits=8),
                        _fmt(r.get("block20_ci95_high"), digits=8),
                    ])
            path = out / f"tab_ablation_spectral_{policy}_{metric_slug}.tex"
            _write_table(
                path,
                caption=f"{metric_label} ({policy}.pt).",
                label=f"tab:ablation-spectral-{policy}-{metric_slug}",
                columns=["Configuration", "Field", "Mean", "CI low", "CI high"],
                rows=rows,
                alignment="llrrr",
                note="Intervals are 95\% circular moving-block CIs with block length 20. Spectral coordinates use the accepted retained-shell index-space evaluator with no spatial smoothing or clipping.",
            )
            policy_children.append(path)
            children.append(path)
        policy_index = out / f"tab_ablation_spectral_{policy}.tex"
        _write_input_index(
            policy_index,
            title=f"Spectral summaries ({policy}.pt), split by metric.",
            label=f"tab:ablation-spectral-{policy}",
            includes=[p.name for p in policy_children],
            note="Low, middle, high and total energy ratios remain beside whole-spectrum LSD in separate compact tables; ratios are reconstruction energy divided by truth energy.",
        )
        children.append(policy_index)
    master = out / "tab_ablation_spectral.tex"
    _write_input_index(master, title="Ablation spectral fidelity.", label="tab:ablation-spectral", includes=[f'tab_ablation_spectral_{policy}.tex' for policy in ('last','best')], note="Low, middle, high and total energy ratios are retained beside whole-spectrum log-spectral distance for every field, stochastic configuration and checkpoint policy.")
    children.insert(0, master)
    return children


def _high_frequency_tables(hf: pd.DataFrame, hf_paired: pd.DataFrame, hann: pd.DataFrame, out: Path) -> list[Path]:
    children: list[Path] = []
    canonical_metrics = [
        ("canonical_shellmean_high_energy_ratio", "Canonical high-band power / truth"),
        ("canonical_spectral_lsd_db", "Canonical whole-spectrum LSD (dB)"),
        ("highband_error_relative_l2", "High-band residual relative $L_2$"),
        ("highband_error_energy_over_truth_high_energy", "Residual energy / truth high energy"),
        ("truth_high_energy_fraction_total", "Truth high energy / total fluctuation"),
        ("reconstruction_high_energy_fraction_total", "Reconstruction high energy / total fluctuation"),
    ]
    for policy in ("last", "best"):
        rows = []
        frame = hf.loc[hf.policy.astype(str).eq(policy)] if not hf.empty and "policy" in hf else pd.DataFrame()
        for method in STOCHASTIC_ORDER:
            for field in FIELD_ORDER:
                values = []
                found = False
                for metric, _ in canonical_metrics:
                    r = _row(frame, method=method, field=field, metric=metric)
                    values.append(_cell(r, digits=7))
                    found = found or r is not None
                if found:
                    rows.append([_display_method(method), field, *values])
        path = out / f"tab_ablation_high_frequency_{policy}.tex"
        _write_table(path, caption=f"Canonical all-field high-frequency diagnostics ({policy}.pt).", label=f"tab:ablation-high-frequency-{policy}", columns=["Configuration", "Field", "Canonical power / truth", "LSD (dB)", "High-band relative $L_2$", "Residual / truth high", "Truth high / total", "Reconstruction high / total"], rows=rows, alignment="llrrrrrr", note="The canonical power ratio integrates shell-mean spectra with the existing trapezoidal weighting; the phase-sensitive residual uses individual complex FFT coefficients over the same strict high-band mask.")
        children.append(path)

    # Mode-sum values are deliberately separate from canonical shell/trapezoid
    # values.  They are needed for the prevalence audit and cannot be used to
    # back-substitute the canonical 3.1199/3.2379 ratios.
    rows = []
    frame = hf.loc[hf.field.astype(str).eq("U1")] if not hf.empty and "field" in hf else pd.DataFrame()
    for policy in ("last", "best"):
        for method in STOCHASTIC_ORDER:
            p = _row(frame, policy=policy, method=method, metric="reconstruction_to_truth_high_energy_ratio")
            e = _row(frame, policy=policy, method=method, metric="highband_error_energy_over_truth_high_energy")
            if p is None and e is None:
                continue
            rows.append([policy, _display_method(method), _cell(p, digits=7), _fmt(p.get("fraction_gt_1"), percent=True) if p is not None else "--", _fmt(p.get("fraction_gt_2"), percent=True) if p is not None else "--", _cell(e, digits=7)])
    path = out / "tab_ablation_high_frequency_modes.tex"
    _write_table(path, caption="U1 mode-sum high-band power and residual audit.", label="tab:ablation-high-frequency-modes", columns=["Policy", "Configuration", "Mode-sum power / truth", "Power $>1$", "Power $>2$", "Residual / truth high"], rows=rows, alignment="llrrrr", note="Mode-sum prevalence is a population summary and is separate from the canonical shell-mean/trapezoidal estimator.")
    children.append(path)

    rows=[]
    for policy in ('last','best'):
        for method in STOCHASTIC_ORDER:
            for metric in ('reconstruction_to_truth_high_energy_ratio','highband_error_relative_l2'):
                r=_row(frame,policy=policy,method=method,metric=metric)
                if r is not None:
                    rows.append([policy,_display_method(method),'Mode-sum power / truth' if metric.startswith('reconstruction') else 'High-band relative L2',*[_fmt(r.get(c)) for c in ['q25','median','q75','p95','p99','min','max']]])
    path=out/'tab_ablation_high_frequency_population_quantiles.tex'
    _write_table(path,caption='U1 state-dispersion quantiles for mode-sum power and high-band residual.',label='tab:ablation-high-frequency-quantiles',columns=['Policy','Configuration','Metric','Q25','Median','Q75','P95','P99','Minimum','Maximum'],rows=rows,note='Population quantiles describe dispersion over 1,000 states; they are not confidence intervals for a mean. Both checkpoint policies and all stochastic variants are retained.')
    children.append(path)
    contrasts=hf_paired.loc[hf_paired.policy.isin(['last','best'])&hf_paired.method.eq('A4')&hf_paired.baseline.eq('A0')&hf_paired.metric.eq('highband_error_relative_l2')]
    rows=[]
    for r in contrasts.itertuples():
        rows.append([r.policy,r.field,_fmt(r.mean_difference),*[_fmt(getattr(r,f'block{b}_ci95_{side}')) for b in [5,20,50] for side in ['low','high']],_fmt(r.fraction_method_less_than_baseline,percent=True)])
    path=out/'tab_ablation_high_frequency_allfield_paired.tex'
    _write_table(path,caption='All-field IID-minus-full high-band relative-L2 contrasts.',label='tab:ablation-high-frequency-allfield-paired',columns=['Policy','Field','Mean difference','Block 5 low','Block 5 high','Block 20 low','Block 20 high','Block 50 low','Block 50 high','Fraction lower'],rows=rows,note='All ten paired differences retain positive interval lower bounds at each of the three circular block lengths. Blocks count held-out states, not original simulation time steps.')
    children.append(path)

    # Hann rows are available in the high-frequency summary for every field;
    # keep the requested U1 comparison and all method rows in one compact file.
    rows = []
    frame = hf.loc[(hf.field.astype(str) == "U1") & hf.metric.astype(str).str.startswith("hann_")] if not hf.empty and "field" in hf else pd.DataFrame()
    hann_metrics = [
        ("hann_highband_error_relative_l2", "Hann high-band relative $L_2$"),
        ("hann_reconstruction_to_truth_high_energy_ratio", "Hann power / truth"),
        ("hann_highband_error_energy_over_truth_total_fluctuation_energy", "Hann residual / total fluctuation"),
        ("hann_truth_high_energy_fraction_total", "Hann truth high / total"),
    ]
    for policy in ("last", "best"):
        for method in STOCHASTIC_ORDER:
            values = []
            found = False
            for metric, _ in hann_metrics:
                r = _row(frame, policy=policy, method=method, metric=metric)
                values.append(_cell(r, digits=7))
                found = found or r is not None
            if found:
                rows.append([policy, _display_method(method), *values])
    # If the evaluator kept Hann rows in a dedicated file, append its exact
    # U1 diagnostic rows as well; this preserves the primary source identity.
    if not hann.empty:
        hframe = hann.loc[hann.field.astype(str).eq("U1")] if "field" in hann else hann
        for policy in ("last", "best"):
            for method in ("A0", "A4"):
                for diagnostic in ("highband_error_relative_l2", "reconstruction_to_truth_high_energy_ratio", "highband_error_energy_over_truth_total_fluctuation_energy"):
                    r = _row(hframe, policy=policy, method=method, window="hann", diagnostic=diagnostic)
                    if r is not None:
                        rows.append([policy, _display_method(method) + " (dedicated Hann source)", diagnostic, _cell(r, digits=7), _fmt(r.get("fraction_gt_1"), percent=True)])
    path = out / "tab_ablation_high_frequency_hann.tex"
    _write_table(path, caption="Hann-window companion diagnostics for high-frequency U1 fidelity.", label="tab:ablation-high-frequency-hann", columns=["Policy", "Configuration", "Hann residual $L_2$", "Hann power / truth", "Hann residual / total", "Hann truth high / total"], rows=rows, alignment="llrrrr", note="Hann values use the separate tapered estimator and are not substituted for the primary untapered canonical quantities.")
    children.append(path)

    rows = []
    for policy in ("last", "best"):
        u1 = hf.loc[(hf.policy.astype(str) == policy) & (hf.field.astype(str) == "U1")] if not hf.empty else pd.DataFrame()
        for method in ("A0", "A4"):
            for metric in ("truth_high_energy_fraction_total", "truth_high_energy_fraction_retained", "reconstruction_high_energy_fraction_total", "reconstruction_high_energy_fraction_retained", "highband_error_energy_over_truth_total_fluctuation_energy", "highband_error_energy_over_truth_retained_fluctuation_energy"):
                r = _row(u1, method=method, metric=metric)
                if r is not None:
                    rows.append([policy, _display_method(method), metric, _cell(r, digits=9)])
    path = out / "tab_ablation_high_frequency_budgets.tex"
    _write_table(path, caption="High-band fluctuation-energy budgets for the U1 prior comparison.", label="tab:ablation-high-frequency-budgets", columns=["Policy", "Configuration", "Metric", "Mean [95\% CI]"], rows=rows, alignment="lllr", note="The truth high-band fraction uses the full centered truth fluctuation denominator. These fractions are not an exact decomposition of the primary physical relative-$L_2$.")
    children.append(path)

    # Include paired prior effects in a separate short table because marginal
    # interval overlap is not a paired comparison.
    rows = []
    if not hf_paired.empty:
        frame = hf_paired.loc[(hf_paired.method.astype(str) == "A4") & (hf_paired.baseline.astype(str) == "A0") & (hf_paired.field.astype(str) == "U1")]
        for policy in ("last", "best"):
            for metric, label in (("canonical_shellmean_high_energy_ratio", "Canonical power / truth"), ("highband_error_relative_l2", "High-band relative $L_2$"), ("reconstruction_to_truth_high_energy_ratio", "Mode-sum power / truth")):
                r = _row(frame, policy=policy, metric=metric)
                if r is not None:
                    rows.append([policy, _raw_tex(label), _effect_cell(r, digits=8), _fmt_pct_change(r.get("relative_mean_change_percent")), _fmt(r.get("fraction_method_less_than_baseline"), percent=True)])
    path = out / "tab_ablation_high_frequency_paired.tex"
    _write_table(path, caption="Paired IID-minus-full high-frequency effects for U1.", label="tab:ablation-high-frequency-paired", columns=["Policy", "Metric", "$\\Delta$ mean [CI]", "Relative change", "Fraction lower"], rows=rows, alignment="llrrr", note="The paired residual effect is the source of the phase-sensitive prior tradeoff; canonical power and mode-sum power remain separate estimators.")
    children.append(path)

    master = out / "tab_ablation_high_frequency.tex"
    _write_input_index(master, title="Ablation high-frequency and spectral-prior diagnostics.", label="tab:ablation-high-frequency", includes=[p.name for p in children], note="Canonical shell/trapezoid, phase-sensitive mode residual, mode-sum prevalence, energy budgets and Hann sensitivity are kept as distinct source metrics.")
    children.insert(0, master)
    return children


def _coupling_tables(coupling: pd.DataFrame, paired: pd.DataFrame, out: Path) -> list[Path]:
    children: list[Path] = []
    metrics = [
        ("joint_pdf_jsd_base2", "Paper JSD (64x64)"),
        ("joint_pdf_jsd_with_overflow_base2", "Overflow JSD (66x66)"),
        ("joint_pdf_reconstruction_retained_fraction", "Predicted retention"),
        ("joint_pdf_truth_retained_fraction", "Truth retention"),
        ("coupling_correlation_abs_error", "Pearson coupling error"),
    ]
    for policy in ("last", "best"):
        rows = []
        frame = coupling.loc[coupling.policy.astype(str).eq(policy)] if not coupling.empty else pd.DataFrame()
        for method in STOCHASTIC_ORDER:
            for target in PAIR_ORDER:
                vals = []
                found = False
                for metric, _ in metrics:
                    r = _row(frame, method=method, metric=metric, target=target)
                    vals.append(_cell(r, digits=7))
                    found = found or r is not None
                if found:
                    rows.append([_display_method(method), target, *vals])
        path = out / f"tab_ablation_coupling_{policy}.tex"
        _write_table(path, caption=f"Coupled-field joint-distribution and retention audit ({policy}.pt).", label=f"tab:ablation-coupling-{policy}", columns=["Configuration", "Pair", "Paper JSD", "Overflow JSD", "Predicted retention", "Truth retention", "Pearson error"], rows=rows, alignment="llrrrrr", note="Paper JSD uses common truth-derived 64$\\times$64 edges with discarded out-of-range mass; overflow JSD uses 66$\\times$66 bins retaining finite tails. Retention is not calibration.")
        children.append(path)

    rows = []
    if not paired.empty:
        frame = paired.loc[(paired.baseline.astype(str) == "A0") & paired.metric.astype(str).isin({m for m, _ in metrics})]
        for policy in ("last", "best"):
            for method in ("A2", "A3", "A5", "A4"):
                for target in PAIR_ORDER:
                    for metric, label in metrics[:2]:
                        r = _row(frame, policy=policy, method=method, metric=metric, target=target)
                        if r is None:
                            continue
                        rows.append([policy, _display_method(method), target, label, _effect_cell(r, digits=7), _fmt_pct_change(r.get("relative_mean_change_percent")), _fmt(r.get("fraction_method_less_than_baseline"), percent=True)])
    path = out / "tab_ablation_coupling_paired.tex"
    _write_table(path, caption="Paired coupling effects relative to the full reference.", label="tab:ablation-coupling-paired", columns=["Policy", "Configuration", "Pair", "Metric", "$\\Delta$ mean [CI]", "Relative change", "Fraction lower"], rows=rows, alignment="lllrllr", note="The paired interval uses the same state identities for each configuration and policy; marginal CIs are not used to infer paired direction.")
    children.append(path)
    master = out / "tab_ablation_coupling.tex"
    _write_input_index(master, title="Coupled-field distribution, retention and correlation audit.", label="tab:ablation-coupling", includes=[p.name for p in children], note="Both histogram definitions, retained fractions, Pearson checks and paired effects are shown for every requested pair and checkpoint policy.")
    children.insert(0, master)
    return children


def _admissibility_tables(adm: pd.DataFrame, counts: pd.DataFrame, events: pd.DataFrame, out: Path) -> list[Path]:
    children: list[Path] = []
    metric_map = [
        ("reconstruction_nonphysical_fraction", "Nonphysical fraction"),
        ("reconstruction_fraction_below_minus_1e4", "Fraction below $-10^{-4}$"),
        ("reconstruction_negative_l2_over_truth_l2", "Negative-part norm / truth norm"),
        ("reconstruction_negative_mean_magnitude", "Negative-part mean magnitude"),
        ("reconstruction_minimum", "Global minimum"),
        ("sensor_max_abs_normalized_error", "Max sensor mismatch"),
    ]
    for policy in ("last", "best"):
        rows = []
        frame = adm.loc[adm.policy.astype(str).eq(policy)] if not adm.empty else pd.DataFrame()
        for method in STOCHASTIC_ORDER:
            for target in ("CH4", "CO", "T", "p"):
                cells = []
                found = False
                for metric, _ in metric_map:
                    r = _row(frame, method=method, metric=metric, target=target)
                    if metric in {"reconstruction_nonphysical_fraction", "reconstruction_fraction_below_minus_1e4"}:
                        cells.append(_cell(r, digits=7))
                    else:
                        cells.append(_cell(r, digits=7))
                    found = found or r is not None
                count = _row(counts.loc[counts.policy.astype(str).eq(policy)] if not counts.empty else counts, method=method, metric="exact_nonphysical_count", target=target)
                if found or count is not None:
                    rows.append([_display_method(method), target, *cells, _fmt(count.get("event_count"), integer=True) if count is not None else "--", _fmt(count.get("total_points"), integer=True) if count is not None else "--", _fmt(count.get("global_minimum"), digits=13) if count is not None else "--"])
        path = out / f"tab_ablation_admissibility_{policy}.tex"
        _write_table(path, caption=f"Unclipped physical admissibility diagnostics ({policy}.pt).", label=f"tab:ablation-admissibility-{policy}", columns=["Configuration", "Field", "Nonphysical fraction", "Below $-10^{-4}$", "Negative norm / truth", "Negative mean", "Mean state minimum","Sensor mismatch", "Exact count", "Total points", "Global minimum"], rows=rows, note="Counts use 40,300,000 points per configuration and policy. Summary intervals describe state means; global minima are separate extrema. Rounded zero percentages do not imply zero rare events.")
        children.append(path)

    rows = []
    if not events.empty:
        frame = events.copy()
        for _, r in frame.sort_values(["policy", "method", "snapshot"], kind="stable").iterrows():
            rows.append([r.get("policy", "--"), _display_method(r.get("method", "--")), r.get("target", "T"), r.get("snapshot", "--"), r.get("time_index", "--"), r.get("event_count", "--"), _fmt(r.get("minimum"),digits=13), r.get("total_points_per_state", "--")])
    path = out / "tab_ablation_admissibility_events.tex"
    _write_table(path, caption="Exact nonpositive-temperature events and affected state/time identities.", label="tab:ablation-admissibility-events", columns=["Policy", "Configuration", "Field", "Snapshot", "Original time index", "Event count", "Minimum", "Points per state"], rows=rows, alignment="lllrrrrr", note="Event identities are retained from the unclipped saved-field audit; no predicted values were clipped before counting or summarizing minima.")
    children.append(path)

    master = out / "tab_ablation_admissibility.tex"
    rows=[[r.policy,_display_method(r.method),r.target,_fmt(r.event_count,integer=True),_fmt(r.total_points,integer=True),_fmt(r.global_minimum,digits=13),r.minimum_state_ids,r.minimum_time_indices] for r in counts.loc[counts.method.isin(STOCHASTIC_ORDER)].itertuples()]
    path=out/'tab_ablation_admissibility_minimum_identities.tex'
    _write_table(path,caption='Exact sign counts and global-minimum state/time identities.',label='tab:ablation-admissibility-minima',columns=['Policy','Configuration','Field','Exact sign count','Total points','Global minimum','Minimum state IDs','Minimum time indices'],rows=rows,note='Species counts use values below zero; temperature and pressure counts use values at or below zero. Global minima are calculated over all 40,300,000 unclipped points, not averaged over state minima.')
    children.append(path)
    _write_input_index(master, title="Physical admissibility and rare-event audit.", label="tab:ablation-admissibility", includes=[p.name for p in children], note="Negative excursions, thresholded sign errors, exact nonpositive T/p counts, minima and affected time identities are retained for both checkpoint policies.")
    children.insert(0, master)
    return children


def _deterministic_tables(summary: pd.DataFrame, paired: pd.DataFrame, out: Path) -> list[Path]:
    children: list[Path] = []
    metrics = [
        ("physical_relative_l2", "Physical relative L2"),
        ("normalized_relative_l2", "Standardized relative L2"),
        ("truth_fluctuation_normalized_l2", "Fluctuation-normalized L2"),
        ("physical_relative_l2_excluding_sensors", "Sensor-excluded physical L2"),
        ("joint_pdf_jsd_base2", "Paper JSD"),
        ("joint_pdf_jsd_with_overflow_base2", "Overflow JSD"),
        ("joint_pdf_reconstruction_retained_fraction", "Predicted retention"),
        ("joint_pdf_truth_retained_fraction", "Truth retention"),
    ]
    rows = []
    for policy in ("last", "best"):
        frame = summary.loc[summary.policy.astype(str).eq(policy)] if not summary.empty else pd.DataFrame()
        for method in ("A0", "A1"):
            for metric, label in metrics:
                targets = (*FIELD_ORDER, "Unobserved_mean") if "l2" in metric else PAIR_ORDER
                for target in targets:
                    r = _row(frame, method=method, metric=metric, target=target)
                    if r is not None:
                        rows.append([policy, _display_method(method), label, target, _cell(r, digits=7)])
    path = out / "tab_deterministic_control_summary.tex"
    _write_table(path, caption="Deterministic objective control versus the stochastic full reference.", label="tab:deterministic-control-summary", columns=["Policy", "Configuration", "Metric", "Target", "Mean [95\% CI]"], rows=rows, alignment="llllr", note="The deterministic objective is a separate control. This table does not imply an ensemble-mean or conditional-diversity comparison.")
    children.append(path)

    rows = []
    if not paired.empty:
        frame = paired.loc[(paired.method.astype(str) == "A1") & (paired.baseline.astype(str) == "A0")]
        for policy in ("last", "best"):
            for metric, label in metrics:
                targets = (*FIELD_ORDER, "Unobserved_mean") if "l2" in metric else PAIR_ORDER
                for target in targets:
                    r = _row(frame, policy=policy, metric=metric, target=target)
                    if r is not None:
                        rows.append([policy, label, target, _effect_cell(r, digits=7), _fmt_pct_change(r.get("relative_mean_change_percent")), _fmt(r.get("fraction_method_less_than_baseline"), percent=True)])
    path = out / "tab_deterministic_control_paired.tex"
    _write_table(path, caption="Paired deterministic-minus-full effects.", label="tab:deterministic-control-paired", columns=["Policy", "Metric", "Target", "$\\Delta$ mean [CI]", "Relative change", "Fraction lower"], rows=rows, alignment="llrllr", note="A negative paired difference means the deterministic control has lower error for that metric; coupling and retention comparisons are reported separately by estimator.")
    children.append(path)
    master = out / "tab_deterministic_control.tex"
    _write_input_index(master, title="Separate deterministic objective control.", label="tab:deterministic-control", includes=[p.name for p in children], note="The control retains both policies, fieldwise and macro reconstruction, pressure/fluctuation checks and coupling/retention companions without entering the stochastic architectural column.")
    children.insert(0, master)
    return children


def _load_inputs(derived: Path) -> dict[str, pd.DataFrame]:
    names = [
        "ablation_provenance", "ablation_all_summary", "ablation_all_paired",
        "ablation_primary_summary", "ablation_primary_paired", "ablation_spectral_summary",
        "ablation_highband_summary", "ablation_highband_paired", "ablation_coupling_audit",
        "ablation_coupling_paired", "ablation_admissibility", "ablation_admissibility_counts",
        "ablation_admissibility_events", "deterministic_control_summary", "deterministic_control_paired",
        "checkpoint_sensitivity",
    ]
    data = {name: _norm(_read(derived / f"{name}.csv")) for name in names}
    # New collectors expose all rows in one source table.  It is the preferred
    # fallback for groups whose specialized table is not present.
    if data["ablation_all_summary"].empty:
        data["ablation_all_summary"] = data["ablation_primary_summary"].copy()
    if data["ablation_all_paired"].empty:
        data["ablation_all_paired"] = data["ablation_primary_paired"].copy()
    if data["ablation_coupling_audit"].empty:
        all_summary = data["ablation_all_summary"]
        data["ablation_coupling_audit"] = all_summary.loc[all_summary.metric.astype(str).str.startswith(("joint_pdf", "coupling_"))].copy()
    if data["ablation_coupling_paired"].empty:
        all_paired = data["ablation_all_paired"]
        data["ablation_coupling_paired"] = all_paired.loc[all_paired.metric.astype(str).str.startswith(("joint_pdf", "coupling_"))].copy()
    if data["deterministic_control_summary"].empty:
        data["deterministic_control_summary"] = data["ablation_all_summary"].loc[data["ablation_all_summary"].method.astype(str).isin(("A0", "A1"))].copy()
    if data["deterministic_control_paired"].empty:
        data["deterministic_control_paired"] = data["ablation_all_paired"].loc[data["ablation_all_paired"].method.astype(str).eq("A1")].copy()
    return data


def _write_latex_index(path: Path, table_dir: Path, children: Mapping[str, Sequence[Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = [
        ("Provenance", "tab_ablation_provenance.tex"),
        ("Reconstruction", "tab_ablation_reconstruction.tex"),
        ("Spectral", "tab_ablation_spectral.tex"),
        ("High-frequency", "tab_ablation_high_frequency.tex"),
        ("Coupling", "tab_ablation_coupling.tex"),
        ("Admissibility", "tab_ablation_admissibility.tex"),
        ("Deterministic control", "tab_deterministic_control.tex"),
    ]
    lines = [
        "% Figure 5 V7 supplementary table entry point.",
        "% Child tables are generated from validated compact CSVs and split by policy/metric.",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{3pt}",
    ]
    # Resolve from the repository root.  The bundle runner compiles the
    # generated report with that working directory, so every child include
    # carries the complete repository-relative path.
    prefix = table_dir.relative_to(REPO_ROOT).as_posix()
    for _, filename in groups:
        lines.append(f"\\input{{{prefix}/{filename}}}")
    lines += ["\\endgroup", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--strict-formal", action="store_true")
    args = parser.parse_args()

    derived = PACKAGE_ROOT / "results" / "derived" / args.timestamp
    out = PACKAGE_ROOT / "docs" / "generated" / args.timestamp / "latex"
    table_dir = out / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale table children before writing a rerun.  This matters when
    # a previously bundled wide table is replaced by the policy/metric split:
    # the delivery directory must contain only the current source-derived
    # package, rather than leaving obsolete ``*.tex`` files for an audit to
    # discover.
    for stale in table_dir.glob("*.tex"):
        stale.unlink()

    data = _load_inputs(derived)
    required = [
        "ablation_provenance", "ablation_all_summary", "ablation_all_paired",
        "ablation_spectral_summary", "ablation_highband_summary",
        "ablation_coupling_audit", "ablation_admissibility_counts",
        "ablation_admissibility_events", "deterministic_control_summary",
    ]
    missing = [name for name in required if data[name].empty]

    generated: list[Path] = []
    generated += _provenance_table(data["ablation_provenance"], table_dir)
    generated += _reconstruction_tables(data["ablation_all_summary"], data["ablation_all_paired"], table_dir)
    generated += _spectral_tables(data["ablation_spectral_summary"], table_dir)
    hann = _read(derived / "ablation_hann_summary.csv")
    generated += _high_frequency_tables(data["ablation_highband_summary"], data["ablation_highband_paired"], _norm(hann), table_dir)
    generated += _coupling_tables(data["ablation_coupling_audit"], data["ablation_coupling_paired"], table_dir)
    generated += _admissibility_tables(data["ablation_admissibility"], data["ablation_admissibility_counts"], data["ablation_admissibility_events"], table_dir)
    generated += _deterministic_tables(data["deterministic_control_summary"], data["deterministic_control_paired"], table_dir)
    _write_latex_index(out / "si_ablation_tables.tex", table_dir, {})

    # Keep a small machine-readable record next to the LaTeX entry point.  It
    # is useful to the parent report and makes a missing compact source clear.
    manifest = {
        "schema_version": "figure5-v7-table-writer-1",
        "timestamp": args.timestamp,
        "status": "pass" if not missing else "blocked",
        "derived_root": str(derived.relative_to(REPO_ROOT)) if derived.exists() else str(derived),
        "required_sources": required,
        "missing_sources": missing,
        "table_count": len(generated),
        "tables": [str(p.relative_to(REPO_ROOT)) for p in sorted(set(generated))],
        "source_rows": {name: int(len(frame)) for name, frame in data.items()},
        "scientific_order": list(STOCHASTIC_ORDER),
        "deterministic_separate": True,
    }
    (out / "table_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "table_count": len(generated), "missing_sources": missing}, indent=2))
    if args.strict_formal and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

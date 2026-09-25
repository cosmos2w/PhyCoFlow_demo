#!/usr/bin/env python
"""Plot fieldwise statewise-CRPS ECDFs from the frozen Figure 5 Cond_T run.

This review candidate reuses processed scores; it performs no inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
import fitz
import numpy as np
import pandas as pd
import yaml

from build_figure5a_statewise_crps import (
    CONFIG, FIELDS, METHODS, REPO_ROOT, RUN, SI_ROOT, load_and_validate, sha256,
)


STEM = "figure5a_fieldwise_crps_ecdf_20260925"
DATA_DIR = SI_ROOT / "results/derived" / STEM
FIGURE_DIR = SI_ROOT / "figures/generated" / STEM
FIGURE_NAME = "fig5a_fieldwise_crps_ecdf"
FIELD_AXES = {
    "Y_CH4": ((0.008, 0.25), (0.01, 0.02, 0.05, 0.1, 0.2)),
    "Y_CO": ((0.04, 0.5), (0.05, 0.1, 0.2, 0.4)),
    "U1": ((0.05, 0.95), (0.05, 0.1, 0.2, 0.4, 0.8)),
    "p": ((0.035, 2.7), (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)),
}
FIELD_LABELS = {
    "Y_CH4": r"$Y_{\mathrm{CH}_4}$",
    "Y_CO": r"$Y_{\mathrm{CO}}$",
    "U1": r"$U_1$",
    "p": r"$p$",
}


def build_ecdf_data(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    for field in FIELDS:
        for method in METHODS:
            group = pairs.loc[pairs.method.eq(method)].copy()
            assert len(group) == 200
            group = group.sort_values([f"crps_{field}", "original_time_index"], kind="stable")
            values = group[f"crps_{field}"].to_numpy(dtype=float)
            assert np.isfinite(values).all() and (values > 0).all()
            limits, _ = FIELD_AXES[field]
            assert float(values.min()) > limits[0] and float(values.max()) < limits[1]
            for rank, record in enumerate(group.itertuples(index=False), start=1):
                rows.append({
                    "field": field,
                    "method": method,
                    "state": int(record.state),
                    "original_time_index": int(record.original_time_index),
                    "draw_count": int(record.draw_count),
                    "normalized_crps": float(getattr(record, f"crps_{field}")),
                    "ecdf_rank": rank,
                    "ecdf_fraction": rank / 200.0,
                })
            summaries.append({
                "field": field,
                "method": method,
                "state_count": 200,
                "draw_count_per_state": 64,
                "mean_normalized_crps": float(np.mean(values)),
                "median_normalized_crps": float(np.median(values)),
                "q1_normalized_crps": float(np.quantile(values, 0.25)),
                "q3_normalized_crps": float(np.quantile(values, 0.75)),
                "min_normalized_crps": float(values.min()),
                "max_normalized_crps": float(values.max()),
            })
    ecdf = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    assert len(ecdf) == 4000 and len(summary) == 20
    assert not ecdf.duplicated(["field", "method", "state"]).any()
    reference = None
    for (field, method), group in ecdf.groupby(["field", "method"], sort=False):
        assert len(group) == 200 and set(group.draw_count) == {64}
        assert np.array_equal(group.ecdf_rank.to_numpy(), np.arange(1, 201))
        assert np.allclose(group.ecdf_fraction, group.ecdf_rank / 200.0, rtol=0, atol=1e-15)
        assert group.normalized_crps.is_monotonic_increasing
        mapping = tuple(sorted(zip(group.state, group.original_time_index)))
        if reference is None:
            reference = mapping
        else:
            assert mapping == reference
    return ecdf, summary


def draw_figure(ecdf: pd.DataFrame, summary: pd.DataFrame, config: dict) -> plt.Figure:
    style = config["style"]
    typography = config["typography_pt"]
    width_mm = float(config["canvas"]["width_mm"])
    height_mm = 109.0
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": float(style["axis_linewidth_pt"]),
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.text(2.2 / width_mm, 104.0 / height_mm, "a", ha="left", va="center",
             fontsize=typography["panel_label"], fontweight="bold", color="#202124")
    fig.text(0.99, 104.0 / height_mm,
             "Fieldwise normalized CRPS  ·  200 paired states  ·  lower is better",
             ha="right", va="center", fontsize=typography["in_plot_annotation"], color="#5C6470")
    fig.text(5.5 / width_mm, 51.5 / height_mm, "Fraction of states ≤ CRPS threshold",
             rotation=90, ha="center", va="center", fontsize=typography["axis_title"], color="#252525")
    fig.text(0.52, 4.0 / height_mm, "Normalized CRPS threshold (log scale)",
             ha="center", va="center", fontsize=typography["axis_title"], color="#252525")

    handles = [Line2D([0], [0], color=style["method_colors"][method], lw=1.3,
                      marker=style["method_markers"][method], markersize=4.3,
                      markerfacecolor="white", markeredgewidth=0.85, label=method)
               for method in METHODS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.52, 97.5 / height_mm),
               bbox_transform=fig.transFigure, ncol=5, frameon=False,
               fontsize=typography["standard_legend"], handlelength=1.7,
               columnspacing=1.65, handletextpad=0.45)

    summary_by_field_method = summary.set_index(["field", "method"])
    positions = ((19.5, 59.5), (102.5, 59.5), (19.5, 13.5), (102.5, 13.5))
    for field, (left_mm, bottom_mm) in zip(FIELDS, positions):
        ax = fig.add_axes([left_mm / width_mm, bottom_mm / height_mm,
                           70.0 / width_mm, 29.0 / height_mm])
        limits, ticks = FIELD_AXES[field]
        ax.set_xscale("log")
        ax.set_xlim(*limits)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{tick:g}" for tick in ticks]))
        ax.minorticks_off()
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
        ax.tick_params(axis="both", labelsize=typography["tick_label"], length=2.0,
                       width=float(style["axis_linewidth_pt"]), pad=2.0, colors="#343A40")
        ax.grid(axis="y", color="#E6E8EC", lw=float(style["grid_linewidth_pt"]), zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#52565C")
        ax.spines[["left", "bottom"]].set_linewidth(float(style["axis_linewidth_pt"]))
        ax.text(0.0, 1.06, FIELD_LABELS[field], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=typography["axis_title"],
                color="#30343A", clip_on=False)

        for method in METHODS:
            group = ecdf.loc[ecdf.field.eq(field) & ecdf.method.eq(method)]
            x = group.normalized_crps.to_numpy(dtype=float)
            fraction = group.ecdf_fraction.to_numpy(dtype=float)
            color = style["method_colors"][method]
            marker = style["method_markers"][method]
            ax.step(np.r_[limits[0], x, limits[1]], np.r_[0.0, fraction, 1.0],
                    where="post", color=color, lw=1.3, alpha=0.98, zorder=2)
            median = float(summary_by_field_method.loc[(field, method), "median_normalized_crps"])
            ax.plot(median, 0.5, marker=marker, markersize=4.3, linestyle="none",
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.9,
                    zorder=4)
    return fig


def write_contract(path: Path, source_dir: Path, width_mm: float) -> None:
    path.write_text(f"""# Figure 5a fieldwise CRPS ECDF candidate

Core claim: normalized CRPS varies by state and physical field; the full statewise distributions can be compared across the five generators without reducing them to one macro average.

Source: `{source_dir.relative_to(REPO_ROOT)}/per_state_method.csv`, plus the formal manifest, draw audit, QA, and CRPS summary. All 200 paired Cond_T states and 64 draws per state are included. No inference was run.

Metric: empirical CRPS is computed at each spatial point using 64 common-normalized samples and the common-normalized truth, then spatially averaged. The four panels show the existing per-state fieldwise reductions for Y_CH4, Y_CO, U1, and p; temperature is conditioned on and excluded. The Figure 5a macro is their equal-weight average.

Marks: each staircase is the empirical cumulative distribution of all 200 statewise field scores for one model. Its height at a given x is the fraction of states with CRPS at or below x. Hollow symbols mark the sample median at 50%; they are not model predictions. Methods retain the manuscript colors and markers.

Axes: y is shared from 0 to 1. Each panel uses an explicitly labeled log-scale x axis with field-specific numeric limits, and all methods within that field share exactly the same scale. The display transform changes no CRPS values or ordering. Proposed insertion width: {width_mm:.0f} mm. This is a separate review candidate and does not modify the current Figure 5 asset.
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=RUN)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert config["schema_version"] == "figure5-art-v3-1"
    pairs, _, manifest, qa = load_and_validate(args.source_run)
    ecdf, summary = build_ecdf_data(pairs)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_dir / "fig5a_fieldwise_crps_ecdf.csv"
    summary_path = args.data_dir / "fig5a_fieldwise_crps_summary.csv"
    ecdf.to_csv(data_path, index=False, float_format="%.17g")
    summary.to_csv(summary_path, index=False, float_format="%.17g")

    plotted = pd.read_csv(data_path)
    assert len(plotted) == 4000 and not plotted.duplicated(["field", "method", "state"]).any()
    for field in FIELDS:
        for method in METHODS:
            group = plotted.loc[plotted.field.eq(field) & plotted.method.eq(method)]
            assert len(group) == 200 and set(group.draw_count) == {64}
            assert np.array_equal(group.ecdf_rank.to_numpy(), np.arange(1, 201))
            assert group.normalized_crps.is_monotonic_increasing
    fig = draw_figure(plotted, summary, config)
    pdf = args.figure_dir / f"{FIGURE_NAME}.pdf"
    svg = args.figure_dir / f"{FIGURE_NAME}.svg"
    png = args.figure_dir / f"{FIGURE_NAME}.png"
    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    with fitz.open(pdf) as document:
        assert len(document) == 1
        page = document[0]
        expected_width = float(config["canvas"]["width_mm"]) / 25.4 * 72.0
        assert abs(page.rect.width - expected_width) < 0.05
        text = page.get_text()
        assert all(method in text for method in METHODS)
        assert "log scale" in text and "200 paired states" in text
        insertion_width_mm = float(config["canvas"]["manuscript_width_mm"])
        scale = insertion_width_mm / float(config["canvas"]["width_mm"])
        pixels_per_point = scale * 300.0 / 72.0
        page.get_pixmap(matrix=fitz.Matrix(pixels_per_point, pixels_per_point),
                        alpha=False).save(args.figure_dir / f"{FIGURE_NAME}_162mm.png")

    write_contract(args.data_dir / "figure_contract.md", args.source_run,
                   float(config["canvas"]["manuscript_width_mm"]))
    provenance = {
        "source_run": str(args.source_run.relative_to(REPO_ROOT)),
        "source_files_sha256": {name: sha256(args.source_run / name) for name in
                                ("per_state_method.csv", "crps_summary.csv",
                                 "method_draw_audit.csv", "manifest.json", "qa.json")},
        "source_formal": manifest["formal"],
        "source_qa": qa["status"],
        "methods": list(METHODS),
        "fields": list(FIELDS),
        "states_per_field_method": 200,
        "draws_per_state": 64,
        "all_state_time_mappings_identical": True,
        "data_file": str(data_path.relative_to(REPO_ROOT)),
        "summary_file": str(summary_path.relative_to(REPO_ROOT)),
        "figure_files": {path.suffix.lstrip("."): str(path.relative_to(REPO_ROOT))
                         for path in (pdf, svg, png)},
    }
    (args.data_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {data_path}")
    print(f"[OK] {summary_path}")
    print(f"[OK] {pdf}")
    print(f"[OK] {png}")


if __name__ == "__main__":
    main()

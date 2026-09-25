#!/usr/bin/env python
"""Render a review-only Figure 5b candidate from the frozen Cond_T UQ table.

This script reads existing statewise reductions. It does not load checkpoints,
generate samples, or change the accepted Figure 5 release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import fitz
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml


SI_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SI_ROOT.parent
METHODS = ("DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT")
FIELDS = ("Y_CH4", "Y_CO", "U1", "p")
RUN = SI_ROOT / "results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6"
CONFIG = SI_ROOT / "configs/figure5_art_v3_1.yaml"
STEM = "figure5b_statewise_spread_error_20260925"
DATA_DIR = SI_ROOT / "results/derived" / STEM
FIGURE_DIR = SI_ROOT / "figures/generated" / STEM
PAIRS_NAME = "fig5b_statewise_pairs.csv"
SUMMARY_NAME = "fig5b_spearman_summary.csv"
FIGURE_NAME = "fig5b_statewise_spread_error"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) & 0x7FFFFFFF


def moving_block_ci(x: np.ndarray, y: np.ndarray, method: str, spec: dict) -> tuple[float, float]:
    """Reproduce the formal runner's state-level temporal block interval."""
    rng = np.random.default_rng(stable_seed(int(spec["seed"]), f"v3|spearman|{method}"))
    n = len(x)
    block = min(int(spec["block_length"]), n)
    samples = np.empty(int(spec["replicates"]), dtype=float)
    for index in range(len(samples)):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        selected = np.concatenate([(np.arange(block) + start) % n for start in starts])[:n]
        samples[index] = float(spearmanr(x[selected], y[selected]).statistic)
    alpha = (1.0 - float(spec["confidence_level"])) / 2.0
    return tuple(float(value) for value in np.nanquantile(samples, [alpha, 1.0 - alpha]))


def load_and_validate(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    qa = json.loads((source_dir / "qa.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "figure5-validation-v3-uq-1"
    assert manifest["status"] == "complete" and manifest["formal"] is True
    assert tuple(manifest["methods"]) == METHODS
    assert len(manifest["states"]) == 200 and len(set(manifest["states"])) == 200
    assert manifest["draws_per_state"] == 64 and manifest["sensor_count"] == 256
    assert qa["status"] == "pass" and qa["paired_state_cohort"] is True
    assert qa["equal_draw_count"] is True and qa["macro_fields"] == list(FIELDS)
    assert qa["macro_weights"] == [0.25] * 4
    assert manifest["bootstrap"]["unit"] == "state"
    assert manifest["bootstrap"]["time_order"] == "original_hdf5_time_index"
    assert manifest["bootstrap"]["method"] == "moving_block_bootstrap"

    source = pd.read_csv(source_dir / "per_state_method.csv")
    reported = pd.read_csv(source_dir / "spread_error_summary.csv")
    audit = pd.read_csv(source_dir / "method_draw_audit.csv")
    assert len(source) == 1000 and len(audit) == 1000
    assert not source.duplicated(["method", "state"]).any()
    assert not audit.duplicated(["method", "state"]).any()
    assert set(zip(source.method, source.state)) == set(zip(audit.method, audit.state))
    assert set(source.draw_count) == set(audit.draw_count) == {64}
    assert set(source.sensor_count) == {256}
    assert audit[["finite", "genuine_stochasticity", "normalization_match"]].all().all()
    assert list(reported.method) == list(METHODS)
    assert set(reported.state_count) == {200}

    numeric = ["macro_normalized_spread", "macro_ensemble_mean_relative_l2"]
    numeric += [f"spread_{field}" for field in FIELDS]
    numeric += [f"error_{field}" for field in FIELDS]
    assert np.isfinite(source[numeric].to_numpy(dtype=float)).all()
    assert (source[numeric] >= 0).all().all()
    assert np.allclose(
        source[[f"spread_{field}" for field in FIELDS]].mean(axis=1),
        source["macro_normalized_spread"], rtol=0, atol=1e-14,
    )
    assert np.allclose(
        source[[f"error_{field}" for field in FIELDS]].mean(axis=1),
        source["macro_ensemble_mean_relative_l2"], rtol=0, atol=1e-14,
    )

    cohort = set(map(int, manifest["states"]))
    reference_mapping = None
    reference_seeds = None
    summaries = []
    for method in METHODS:
        group = source.loc[source.method.eq(method)].sort_values("original_time_index")
        assert len(group) == 200 and set(map(int, group.state)) == cohort
        assert group.original_time_index.is_unique
        mapping = tuple(sorted(zip(group.state.astype(int), group.original_time_index.astype(int))))
        seeds = tuple(sorted(zip(group.state.astype(int), group.generation_seed_first.astype(int), group.generation_seed_last.astype(int))))
        if reference_mapping is None:
            reference_mapping, reference_seeds = mapping, seeds
        else:
            assert mapping == reference_mapping and seeds == reference_seeds
        for state, first, last in seeds:
            assert first == stable_seed(20260830, "generation", "U2", state, 0)
            assert last == stable_seed(20260830, "generation", "U2", state, 63)

        spread = group.macro_normalized_spread.to_numpy(dtype=float)
        error = group.macro_ensemble_mean_relative_l2.to_numpy(dtype=float)
        rho = float(spearmanr(spread, error).statistic)
        ci_low, ci_high = moving_block_ci(spread, error, method, manifest["bootstrap"])
        expected = reported.set_index("method").loc[method]
        assert abs(rho - float(expected.spearman_rho)) < 1e-12
        assert abs(ci_low - float(expected.spearman_ci_low)) < 1e-12
        assert abs(ci_high - float(expected.spearman_ci_high)) < 1e-12
        summaries.append({
            "method": method,
            "state_count": len(group),
            "draw_count_per_state": 64,
            "spearman_rho": rho,
            "spearman_ci_low": ci_low,
            "spearman_ci_high": ci_high,
            "bootstrap_method": "temporal_moving_block_state_resampling",
            "bootstrap_block_length": int(manifest["bootstrap"]["block_length"]),
            "bootstrap_replicates": int(manifest["bootstrap"]["replicates"]),
        })

    columns = ["method", "state", "original_time_index", "sensor_count", "draw_count",
               "generation_seed_first", "generation_seed_last", *numeric]
    pairs = source[columns].copy()
    pairs["method"] = pd.Categorical(pairs.method, categories=METHODS, ordered=True)
    pairs = pairs.sort_values(["method", "original_time_index"]).reset_index(drop=True)
    pairs["method"] = pairs.method.astype(str)
    return pairs, pd.DataFrame(summaries), manifest, qa


def quintile_medians(spread: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(spread, kind="stable")
    bins = np.array_split(order, 5)
    assert all(len(group) == 40 for group in bins)
    return (
        np.array([np.median(spread[group]) for group in bins]),
        np.array([np.median(error[group]) for group in bins]),
    )


def draw_figure(pairs: pd.DataFrame, summary: pd.DataFrame, config: dict) -> plt.Figure:
    typography = config["typography_pt"]
    style = config["style"]
    width_mm = float(config["canvas"]["width_mm"])
    height_mm = 61.0
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
    left_mm, right_mm, gap_mm = 17.5, 3.0, 3.8
    bottom_mm, top_mm = 13.0, 45.2
    panel_width_mm = (width_mm - left_mm - right_mm - 4 * gap_mm) / 5
    rho_by_method = summary.set_index("method")

    fig.text(2.2 / width_mm, 57.0 / height_mm, "b", ha="left", va="center",
             fontsize=typography["panel_label"], fontweight="bold", color="#202124")
    fig.text(5.5 / width_mm, 29.2 / height_mm,
             "Ensemble-mean\nrelative $L_2$ error", rotation=90,
             ha="center", va="center", fontsize=typography["axis_title"], color="#252525")
    fig.text(0.52, 4.3 / height_mm, "Normalized ensemble spread",
             ha="center", va="center", fontsize=typography["axis_title"], color="#252525")
    fig.text(0.99, 57.1 / height_mm, "200 states / method  ·  dots: states  ·  line: spread-quintile medians",
             ha="right", va="center", fontsize=typography["in_plot_annotation"], color="#5C6470")

    for index, method in enumerate(METHODS):
        x_mm = left_mm + index * (panel_width_mm + gap_mm)
        ax = fig.add_axes([x_mm / width_mm, bottom_mm / height_mm,
                           panel_width_mm / width_mm, (top_mm - bottom_mm) / height_mm])
        group = pairs.loc[pairs.method.eq(method)]
        spread = group.macro_normalized_spread.to_numpy(dtype=float)
        error = group.macro_ensemble_mean_relative_l2.to_numpy(dtype=float)
        color = style["method_colors"][method]
        ax.scatter(spread, error, s=5.5, color=color, alpha=0.34,
                   linewidths=0, rasterized=False, zorder=2)
        x_med, y_med = quintile_medians(spread, error)
        ax.plot(x_med, y_med, color=color, lw=1.05, marker="o", markersize=2.0,
                markeredgewidth=0, zorder=3)

        x_span = float(spread.max() - spread.min())
        y_span = float(error.max() - error.min())
        ax.set_xlim(max(0.0, float(spread.min()) - 0.055 * x_span),
                    float(spread.max()) + 0.055 * x_span)
        ax.set_ylim(max(0.0, float(error.min()) - 0.065 * y_span),
                    float(error.max()) + 0.065 * y_span)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", labelsize=typography["tick_label"],
                       length=2.0, width=float(style["axis_linewidth_pt"]),
                       pad=2.0, colors="#343A40")
        ax.grid(axis="y", color="#E6E8EC", lw=float(style["grid_linewidth_pt"]), zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#52565C")
        ax.spines[["left", "bottom"]].set_linewidth(float(style["axis_linewidth_pt"]))
        ax.text(0.0, 1.175, method, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=typography["axis_title"], color=color,
                fontweight="bold" if method == "DMF-Gen" else "medium", clip_on=False)
        ax.text(0.0, 1.055, f"Spearman ρ = {rho_by_method.loc[method, 'spearman_rho']:.3f}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=typography["in_plot_annotation"], color="#444A52", clip_on=False)
    return fig


def write_contract(path: Path, source_dir: Path, width_mm: float) -> None:
    path.write_text(f"""# Figure 5b statewise candidate

Core claim: the association between normalized ensemble spread and reconstruction difficulty varies across the five generators and is strongest for DMF-Gen.

Archetype: quantitative grid, five equal method panels in a full-width strip. Proposed insertion width: {width_mm:.0f} mm; each panel has its own labeled linear x and y limits so all 200 states remain visible. Spearman rho is scale invariant. This candidate requires a Figure 5 layout change before replacement of the current panel b.

Source: `{source_dir.relative_to(REPO_ROOT)}/per_state_method.csv`, plus that run's formal manifest, QA, draw audit, and spread/error summary. No inference was run. The same 200 paired Cond_T states and 64 draws per state are used for all five methods.

Metrics: for each state and each of Y_CH4, Y_CO, U1, and p, spread is the spatial RMS of the 64-draw sample standard deviation (ddof=1) after common training-statistic normalization. Ensemble-mean error is the relative L2 norm of the mean prediction in physical units. The x and y values each average their four fieldwise values with weight 0.25. Temperature is the conditioned field and is excluded from these macros.

Marks: every state is shown as a transparent point. The line connects medians of five equal-count spread bins (40 states each), sorted within method. It is a descriptive summary, not a regression fit. No points are removed. Each panel uses linear axes and its own stated numeric limits.

Statistics: Spearman rho is calculated from the plotted pairs. The 95% intervals reproduce the formal 2,000-replicate temporal moving-block bootstrap, resampling states in original time order with block length 25. The intervals are recorded in `fig5b_spearman_summary.csv`; no significance test is added to the panel.

Interpretation: spread-based ranking of difficult states is an association. These figures do not establish calibration, physical causality, or prospective prediction. Method-specific axis limits help show each cloud and should not be read as equal effect sizes.
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
    pairs, summary, manifest, qa = load_and_validate(args.source_run)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.data_dir / PAIRS_NAME
    summary_path = args.data_dir / SUMMARY_NAME
    pairs.to_csv(pairs_path, index=False, float_format="%.17g")
    summary.to_csv(summary_path, index=False, float_format="%.17g")

    plotted = pd.read_csv(pairs_path)
    assert len(plotted) == 1000 and not plotted.duplicated(["method", "state"]).any()
    for method in METHODS:
        group = plotted.loc[plotted.method.eq(method)]
        expected = summary.set_index("method").loc[method, "spearman_rho"]
        observed = float(spearmanr(group.macro_normalized_spread,
                                   group.macro_ensemble_mean_relative_l2).statistic)
        assert len(group) == 200 and abs(observed - expected) < 1e-12

    figure = draw_figure(plotted, summary, config)
    pdf = args.figure_dir / f"{FIGURE_NAME}.pdf"
    svg = args.figure_dir / f"{FIGURE_NAME}.svg"
    png = args.figure_dir / f"{FIGURE_NAME}.png"
    figure.savefig(pdf)
    figure.savefig(svg)
    figure.savefig(png, dpi=600)
    plt.close(figure)

    with fitz.open(pdf) as document:
        assert len(document) == 1
        page = document[0]
        expected_width = float(config["canvas"]["width_mm"]) / 25.4 * 72.0
        assert abs(page.rect.width - expected_width) < 0.05
        text = page.get_text()
        assert all(method in text for method in METHODS)
        assert text.count("Spearman") == 5
        insertion_width_mm = float(config["canvas"]["manuscript_width_mm"])
        scale = insertion_width_mm / float(config["canvas"]["width_mm"])
        pixels_per_point = scale * 300.0 / 72.0
        preview = page.get_pixmap(matrix=fitz.Matrix(pixels_per_point, pixels_per_point),
                                  alpha=False)
        preview.save(args.figure_dir / f"{FIGURE_NAME}_162mm.png")

    write_contract(args.data_dir / "figure_contract.md", args.source_run,
                   float(config["canvas"]["manuscript_width_mm"]))
    provenance = {
        "source_run": str(args.source_run.relative_to(REPO_ROOT)),
        "source_files_sha256": {name: sha256(args.source_run / name) for name in
                                ("per_state_method.csv", "spread_error_summary.csv",
                                 "method_draw_audit.csv", "manifest.json", "qa.json")},
        "source_formal": manifest["formal"],
        "source_qa": qa["status"],
        "methods": list(METHODS),
        "states_per_method": 200,
        "draws_per_state": 64,
        "fields": list(FIELDS),
        "rho_reproduced": True,
        "bootstrap_ci_reproduced": True,
        "all_state_time_mappings_identical": True,
        "data_file": str(pairs_path.relative_to(REPO_ROOT)),
        "summary_file": str(summary_path.relative_to(REPO_ROOT)),
        "figure_files": {path.suffix.lstrip("."): str(path.relative_to(REPO_ROOT))
                         for path in (pdf, svg, png)},
    }
    (args.data_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {pairs_path}")
    print(f"[OK] {summary_path}")
    print(f"[OK] {pdf}")
    print(f"[OK] {png}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build a review-only, statewise Figure 5a normalized-CRPS candidate.

Reuses the frozen formal Cond_T reductions; no checkpoint or inference is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import fitz
import numpy as np
import pandas as pd
import yaml


SI_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SI_ROOT.parent
METHODS = ("DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT")
FIELDS = ("Y_CH4", "Y_CO", "U1", "p")
RUN = SI_ROOT / "results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6"
CONFIG = SI_ROOT / "configs/figure5_art_v3_1.yaml"
STEM = "figure5a_statewise_crps_20260925"
DATA_DIR = SI_ROOT / "results/derived" / STEM
FIGURE_DIR = SI_ROOT / "figures/generated" / STEM
FIGURE_NAME = "fig5a_statewise_normalized_crps"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) & 0x7FFFFFFF


def bootstrap_mean(values: np.ndarray, method: str, spec: dict) -> tuple[float, float]:
    """Match the formal runner's original-time-order moving-block bootstrap."""
    rng = np.random.default_rng(stable_seed(int(spec["seed"]), f"v3|crps|{method}"))
    n = len(values)
    block = min(int(spec["block_length"]), n)
    samples = np.empty(int(spec["replicates"]), dtype=float)
    for index in range(len(samples)):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        selected = np.concatenate([(np.arange(block) + start) % n for start in starts])[:n]
        samples[index] = float(np.mean(values[selected]))
    alpha = (1.0 - float(spec["confidence_level"])) / 2.0
    return tuple(float(value) for value in np.quantile(samples, [alpha, 1.0 - alpha]))


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
    reported = pd.read_csv(source_dir / "crps_summary.csv")
    audit = pd.read_csv(source_dir / "method_draw_audit.csv")
    assert len(source) == len(audit) == 1000
    assert not source.duplicated(["method", "state"]).any()
    assert not audit.duplicated(["method", "state"]).any()
    assert set(zip(source.method, source.state)) == set(zip(audit.method, audit.state))
    assert set(source.draw_count) == set(audit.draw_count) == {64}
    assert set(source.sensor_count) == {256}
    assert audit[["finite", "genuine_stochasticity", "normalization_match"]].all().all()
    assert list(reported.method) == list(METHODS)
    assert set(reported.state_count) == {200}
    assert set(reported.field_macro_weight) == {0.25}

    numeric = ["macro_normalized_crps", *(f"crps_{field}" for field in FIELDS)]
    assert np.isfinite(source[numeric].to_numpy(dtype=float)).all()
    assert (source[numeric] >= 0).all().all()
    assert np.allclose(source[[f"crps_{field}" for field in FIELDS]].mean(axis=1),
                       source.macro_normalized_crps, rtol=0, atol=1e-14)

    cohort = set(map(int, manifest["states"]))
    reference_mapping = None
    summaries = []
    reported_by_method = reported.set_index("method")
    for method in METHODS:
        group = source.loc[source.method.eq(method)].sort_values("original_time_index")
        assert len(group) == 200 and set(map(int, group.state)) == cohort
        assert group.original_time_index.is_unique
        mapping = tuple(sorted(zip(group.state.astype(int), group.original_time_index.astype(int))))
        if reference_mapping is None:
            reference_mapping = mapping
        else:
            assert mapping == reference_mapping
        for row in group.itertuples():
            assert row.generation_seed_first == stable_seed(20260830, "generation", "U2", row.state, 0)
            assert row.generation_seed_last == stable_seed(20260830, "generation", "U2", row.state, 63)

        crps = group.macro_normalized_crps.to_numpy(dtype=float)
        mean = float(np.mean(crps))
        low, high = bootstrap_mean(crps, method, manifest["bootstrap"])
        expected = reported_by_method.loc[method]
        assert abs(mean - float(expected.mean_normalized_crps)) < 1e-12
        assert abs(low - float(expected.crps_ci_low)) < 1e-12
        assert abs(high - float(expected.crps_ci_high)) < 1e-12
        summaries.append({
            "method": method,
            "state_count": 200,
            "draw_count_per_state": 64,
            "mean_normalized_crps": mean,
            "crps_ci_low": low,
            "crps_ci_high": high,
            "median_normalized_crps": float(np.median(crps)),
            "q1_normalized_crps": float(np.quantile(crps, 0.25)),
            "q3_normalized_crps": float(np.quantile(crps, 0.75)),
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


def draw_figure(pairs: pd.DataFrame, summary: pd.DataFrame, config: dict) -> plt.Figure:
    typography = config["typography_pt"]
    style = config["style"]
    width_mm = float(config["canvas"]["width_mm"])
    height_mm = 75.0
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
    ax = fig.add_axes([31.0 / width_mm, 12.5 / height_mm,
                       119.0 / width_mm, 51.0 / height_mm])
    fig.text(2.2 / width_mm, 70.0 / height_mm, "a", ha="left", va="center",
             fontsize=typography["panel_label"], fontweight="bold", color="#202124")
    fig.text(0.99, 70.0 / height_mm,
             "200 states / method  ·  64 draws / state  ·  mean and 95% CI",
             ha="right", va="center", fontsize=typography["in_plot_annotation"], color="#5C6470")
    fig.text(0.83, 65.0 / height_mm, "Mean [95% CI]", ha="left", va="center",
             fontsize=typography["in_plot_annotation"], color="#444A52")
    fig.text(0.505, 4.0 / height_mm, "Normalized CRPS (log scale; lower is better)",
             ha="center", va="center", fontsize=typography["axis_title"], color="#252525")

    ax.set_xscale("log")
    ax.set_xlim(0.04, 1.0)
    ax.set_ylim(-0.48, 4.48)
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.4, 0.8]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.05", "0.1", "0.2", "0.4", "0.8"]))
    ax.minorticks_off()
    ax.set_yticks([4, 3, 2, 1, 0], METHODS)
    ax.tick_params(axis="x", labelsize=typography["tick_label"], length=2.0,
                   width=float(style["axis_linewidth_pt"]), pad=2.0, colors="#343A40")
    ax.tick_params(axis="y", labelsize=typography["tick_label"], length=0,
                   pad=5.0, colors="#343A40")
    ax.grid(axis="x", color="#E6E8EC", lw=float(style["grid_linewidth_pt"]), zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#52565C")
    ax.spines["bottom"].set_linewidth(float(style["axis_linewidth_pt"]))

    summary_by_method = summary.set_index("method")
    for index, method in enumerate(METHODS):
        y = 4 - index
        color = style["method_colors"][method]
        marker = style["method_markers"][method]
        group = pairs.loc[pairs.method.eq(method)]
        values = group.macro_normalized_crps.to_numpy(dtype=float)
        assert len(values) == 200
        rng = np.random.default_rng(stable_seed(20260925, "figure5a", method))
        jitter = rng.uniform(-0.30, 0.00, size=len(values))
        ax.scatter(values, y + jitter, s=5.0, color=color, alpha=0.43,
                   linewidths=0, rasterized=False, zorder=2)
        mean = float(summary_by_method.loc[method, "mean_normalized_crps"])
        low = float(summary_by_method.loc[method, "crps_ci_low"])
        high = float(summary_by_method.loc[method, "crps_ci_high"])
        ax.errorbar(mean, y + 0.17, xerr=np.array([[mean - low], [high - mean]]),
                    fmt=marker, markersize=style["comparison_marker_size_pt"],
                    markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=style["comparison_marker_edge_width_pt"],
                    ecolor=color, elinewidth=1.15, capsize=2.3, capthick=1.0,
                    zorder=4)
        fig.text(0.83, (12.5 + 51.0 * (y + 0.48) / 4.96) / height_mm,
                 f"{mean:.3f}  [{low:.3f}, {high:.3f}]", ha="left", va="center",
                 fontsize=typography["in_plot_annotation"], color=color)
    return fig


def write_contract(path: Path, source_dir: Path, width_mm: float) -> None:
    path.write_text(f"""# Figure 5a statewise normalized-CRPS candidate

Core claim: the five generators have different statewise normalized-CRPS distributions in the same 200-state Cond_T cohort. Lower CRPS is better.

Source: `{source_dir.relative_to(REPO_ROOT)}/per_state_method.csv`, plus the formal manifest, draw audit, QA, and CRPS summary. No inference was run. All methods use 200 paired states and 64 draws per state.

Metric: empirical CRPS is calculated pointwise from 64 common-normalized physical-unit draws and the common-normalized truth, then averaged over space. The macro is the equal-weight mean across Y_CH4, Y_CO, U1, and p. Temperature is conditioned on and excluded.

Marks: all 200 statewise macro values per method are shown as transparent dots with deterministic vertical jitter only. Hollow symbols and whiskers show the arithmetic mean and the formal 95% moving-block state-bootstrap interval (2,000 replicates, block length 25, original time order). Numeric means and intervals are repeated at right. No observations are omitted.

Axis: the shared horizontal CRPS axis is logarithmic and explicitly labeled; this makes the low-error DMF-Gen and SiT distributions readable alongside the larger CRPS values. The log display does not change the values, arithmetic means, or confidence intervals. Proposed insertion width: {width_mm:.0f} mm. The present Figure 5a asset remains unchanged.
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
    pairs_path = args.data_dir / "fig5a_statewise_crps.csv"
    summary_path = args.data_dir / "fig5a_crps_summary.csv"
    pairs.to_csv(pairs_path, index=False, float_format="%.17g")
    summary.to_csv(summary_path, index=False, float_format="%.17g")

    plotted = pd.read_csv(pairs_path)
    assert len(plotted) == 1000 and not plotted.duplicated(["method", "state"]).any()
    for method in METHODS:
        values = plotted.loc[plotted.method.eq(method), "macro_normalized_crps"].to_numpy(dtype=float)
        expected = summary.set_index("method").loc[method, "mean_normalized_crps"]
        assert len(values) == 200 and abs(float(np.mean(values)) - expected) < 1e-12

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
        assert "log scale" in text and "Mean [95% CI]" in text
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
        "states_per_method": 200,
        "draws_per_state": 64,
        "fields": list(FIELDS),
        "source_mean_ci_reproduced": True,
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

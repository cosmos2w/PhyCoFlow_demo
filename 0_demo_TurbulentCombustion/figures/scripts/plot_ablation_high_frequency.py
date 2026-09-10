#!/usr/bin/env python
"""Plot cache-derived Cond_T high-frequency diagnostics.

The companion analyzer writes all statistics used here.  This script only
reads those CSV/JSON artifacts and creates an editable three-panel figure:
population U1 high-band energy ratios, phase-sensitive high-band residuals,
and shell-resolved ratios for A0/A1/A4.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEMO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = DEMO / "Save_TrainedModel/ablation_condT/evaluation_20260910/high_frequency"
DEFAULT_OUTPUT = DEMO / "figures/generated/ablation_condT_high_frequency_20260910"
METHODS = tuple(f"A{i}" for i in range(6))
POLICIES = ("last", "best")
METHOD_LABELS = {
    "A0": "A0 (new checkpoint)",
    "A1": "A1",
    "A2": "A2",
    "A3": "A3",
    "A4": "A4",
    "A5": "A5",
}
COLORS = {
    "A0": "#1B4F72",
    "A1": "#D55E00",
    "A2": "#009E73",
    "A3": "#CC79A7",
    "A4": "#7B3294",
    "A5": "#E69F00",
}
POLICY_OFFSETS = {"last": -0.18, "best": 0.18}
POLICY_MARKERS = {"last": "o", "best": "s"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing high-frequency artifact: {path}. Run "
            "src/analyze_ablation_high_frequency.py first."
        )


def read_inputs(input_dir: Path):
    summary_path = input_dir / "summary_high_frequency.csv"
    per_state_path = input_dir / "per_state_high_frequency.csv"
    spectra_path = input_dir / "population_U1_spectra.csv"
    metadata_path = input_dir / "analysis_metadata.json"
    hann_path = input_dir / "U1_hann_robustness_summary.csv"
    for path in (summary_path, per_state_path, spectra_path, metadata_path, hann_path):
        require_file(path)

    summary = pd.read_csv(summary_path)
    per_state = pd.read_csv(
        per_state_path,
        usecols=["policy", "method", "snapshot", "field", "metric", "value"],
    )
    spectra = pd.read_csv(spectra_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hann = pd.read_csv(hann_path)
    expected_pairs = {(policy, method) for policy in POLICIES for method in METHODS}
    actual_pairs = set(zip(summary["policy"], summary["method"]))
    if actual_pairs != expected_pairs:
        raise ValueError(f"Summary lacks policy/method pairs: {sorted(expected_pairs - actual_pairs)}")
    required_metrics = {
        "reconstruction_to_truth_high_energy_ratio",
        "highband_error_relative_l2",
        "canonical_shellmean_high_energy_ratio",
        "canonical_spectral_lsd_db",
    }
    if not required_metrics.issubset(set(summary["metric"])):
        raise ValueError(f"Summary lacks required high-frequency metrics: {required_metrics - set(summary['metric'])}")
    u1_summary = summary[summary["field"].eq("U1")]
    if set(u1_summary["n"].astype(int)) != {1000}:
        raise ValueError("The plotted U1 summary is not a complete 1,000-snapshot population")
    u1_per_state = per_state[per_state["field"].eq("U1")]
    for metric in ("reconstruction_to_truth_high_energy_ratio", "highband_error_relative_l2"):
        subset = u1_per_state[u1_per_state["metric"].eq(metric)]
        if len(subset) != len(POLICIES) * len(METHODS) * 1000:
            raise ValueError(f"Per-state U1 coverage is incomplete for {metric}: {len(subset)} rows")
    required_spectrum_columns = {
        "policy", "method", "field", "shell_index", "wavenumber", "kmax",
        "high_band", "n", "median", "q25", "q75", "p95",
    }
    if not required_spectrum_columns.issubset(spectra.columns):
        raise ValueError(f"Population spectrum artifact lacks {required_spectrum_columns - set(spectra.columns)}")
    if not set(spectra["field"]).issubset({"U1"}):
        raise ValueError("Population spectrum artifact contains an unexpected field")
    if set(spectra["n"].astype(int)) != {1000}:
        raise ValueError("Population shell summaries are not based on 1,000 snapshots")
    if metadata.get("grid", {}).get("high_band_rule", "").find("> 2/3") < 0:
        raise ValueError("Metadata does not record the strict k > 2/3*kmax high-band rule")
    return summary, per_state, spectra, metadata, hann


def summary_row(summary: pd.DataFrame, policy: str, method: str, metric: str):
    row = summary[
        summary["policy"].eq(policy)
        & summary["method"].eq(method)
        & summary["field"].eq("U1")
        & summary["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one summary row for {policy}/{method}/{metric}, found {len(row)}")
    return row.iloc[0]


def add_interval_panel(
    ax,
    summary: pd.DataFrame,
    metric: str,
    *,
    ylabel: str,
    title: str,
    yscale: str = "log",
    ideal: float | None = 1.0,
    reference_label: str | None = None,
):
    for method_index, method in enumerate(METHODS):
        for policy in POLICIES:
            row = summary_row(summary, policy, method, metric)
            x = method_index + POLICY_OFFSETS[policy]
            median = float(row["median"])
            lower = max(median - float(row["q25"]), 1e-15)
            upper = max(float(row["q75"]) - median, 1e-15)
            ax.errorbar(
                [x], [median], yerr=[[lower], [upper]],
                fmt=POLICY_MARKERS[policy], color=COLORS[method],
                markersize=4.7, markeredgecolor="white", markeredgewidth=0.45,
                elinewidth=1.05, capsize=2.2, zorder=3,
            )
            ax.scatter(
                [x], [float(row["p95"])], marker="^", s=20, color=COLORS[method],
                edgecolor="white", linewidth=0.35, zorder=4,
            )
    if ideal is not None:
        ax.axhline(ideal, color="0.35", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
        if reference_label:
            ax.text(
                0.99, ideal, reference_label, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.4, color="0.35",
            )
    ax.set_yscale(yscale)
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.set_xticks(range(len(METHODS)), METHODS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.grid(axis="y", which="major", color="0.87", linewidth=0.6)
    ax.grid(axis="y", which="minor", color="0.94", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.99, 0.03, "IQR bars; triangles p95",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.7, color="0.32",
    )


def add_spectrum_panel(ax, spectra: pd.DataFrame):
    for method in ("A0", "A1", "A4"):
        for policy in POLICIES:
            subset = spectra[spectra["policy"].eq(policy) & spectra["method"].eq(method)].sort_values("shell_index")
            if subset.empty:
                raise ValueError(f"Missing shell population for {policy}/{method}")
            x = subset["wavenumber"].to_numpy(float) / subset["kmax"].to_numpy(float)
            median = subset["median"].to_numpy(float)
            q25 = subset["q25"].to_numpy(float)
            q75 = subset["q75"].to_numpy(float)
            color = COLORS[method]
            line_style = "-" if policy == "last" else "--"
            valid = np.isfinite(x) & np.isfinite(median) & (median > 0)
            ax.plot(
                x[valid], median[valid], color=color, linestyle=line_style,
                linewidth=1.25, label=f"{method} {policy}", zorder=3,
            )
            band_valid = valid & np.isfinite(q25) & np.isfinite(q75) & (q25 > 0) & (q75 > 0)
            ax.fill_between(x[band_valid], q25[band_valid], q75[band_valid], color=color, alpha=0.10, linewidth=0)
    ax.axvspan(2.0 / 3.0, 1.0, color="0.5", alpha=0.08, zorder=0)
    ax.axhline(1.0, color="0.35", linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(0.675, 0.97, "strict HF band", transform=ax.transAxes, fontsize=6.8, color="0.35", va="top")
    ax.set_xlim(0.0, 1.01)
    ax.set_yscale("log")
    ax.set_xlabel(r"retained shell wavenumber / $k_{\max}$ (index space)")
    ax.set_ylabel("shell-mean spectrum ratio\n(reconstruction / truth)")
    ax.set_title("(c) U1 shell population: A0, A1, and A4", loc="left", fontweight="bold", pad=6)
    ax.grid(axis="both", which="major", color="0.87", linewidth=0.6)
    ax.grid(axis="y", which="minor", color="0.94", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.99, 0.03, "lines median; ribbons IQR",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.7, color="0.32",
    )


def write_contract(output_dir: Path, input_dir: Path, metadata: dict):
    contract = f"""# Cond_T high-frequency diagnostic figure contract

## Core scientific claim

Across the held-out 1,000-snapshot population, the U1 high-band diagnostic
separates spectral energy surplus from phase-sensitive high-band residual error.
The figure is an exploratory diagnostic for the A4 high-frequency behavior;
it does not change the physical ablation metrics.

## Source files

- Evaluation artifact directory: `{input_dir}`
- `summary_high_frequency.csv` (all six methods, both saved-checkpoint policies)
- `per_state_high_frequency.csv` (per-snapshot values for all requested fields)
- `population_U1_spectra.csv` (per-shell population summaries)
- `U1_hann_robustness_summary.csv` (compact no-window/Hann companion table)
- `analysis_metadata.json` (grid, shell, normalization, and validation record)

## Panel map

- **(a)** U1 reconstruction/truth high-band mode-energy ratio. Circles/squares
  encode last/best saved checkpoints; bars are the snapshot IQR and triangles
  are p95. The dashed line is the ideal ratio of 1.
- **(b)** U1 phase-sensitive high-band residual RMS divided by truth high-band
  RMS, using individual Fourier-mode sums. Bars and triangles use the same
  population summaries as panel (a); the line at 1 marks equal residual and
  truth high-band energy.
- **(c)** Population median shell-mean spectrum ratio with IQR ribbons for
  A0, A1, and A4. The shaded region is the strict high band, `k > 2/3 kmax`.

## Definitions and caveats

- FFTs use the canonical structured-grid ordering and isotropic retained-shell
  cutoff from `common.spectral`; wavenumbers are topological/index-space
  quantities because the recovered mesh is treated conservatively.
- Primary high-band metrics are mode-sum energies. They are distinct from the
  canonical shell-mean/trapezoidal high-energy ratio retained for parity with
  the main evaluator.
- Full fluctuation denominators sum all centered FFT modes and satisfy Parseval;
  retained denominators exclude shell 0, under-populated shells, and cutoff
  corners. No smoothing, clipping, or spatial-pixel independence assumption is
  used.
- The Hann rows are a boundary-leakage sensitivity check with mean-square
  window correction. The no-window values remain the primary diagnostic.
- Population intervals use quantiles and circular block bootstrap summaries;
  matched block-20 uncertainty intervals are in `paired_high_frequency.csv`.

## Reproduction

```text
python \\
  0_demo_TurbulentCombustion/figures/scripts/plot_ablation_high_frequency.py \\
  --input-dir {input_dir} --output-dir {output_dir}
```
"""
    (output_dir / "figure_contract.md").write_text(contract, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, per_state, spectra, metadata, hann = read_inputs(input_dir)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.55), constrained_layout=False)
    add_interval_panel(
        axes[0], summary, "reconstruction_to_truth_high_energy_ratio",
        ylabel="HF reconstruction / truth mode energy",
        title="(a) U1 high-band energy balance",
        reference_label="ideal ratio = 1",
    )
    add_interval_panel(
        axes[1], summary, "highband_error_relative_l2",
        ylabel="HF residual RMS / truth HF RMS",
        title="(b) U1 phase-sensitive HF residual",
        reference_label="unity: residual = truth HF",
    )
    add_spectrum_panel(axes[2], spectra)
    axes[0].set_xlabel("ablation method")
    axes[1].set_xlabel("ablation method")

    method_handles = [
        Line2D([0], [0], color=COLORS[method], linewidth=2.0, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    policy_handles = [
        Line2D([0], [0], color="0.2", marker=POLICY_MARKERS[policy], linestyle="None", markersize=5.0, label=f"{policy} checkpoint")
        for policy in POLICIES
    ]
    p95_handle = Line2D([0], [0], color="0.2", marker="^", linestyle="None", markersize=5.0, label="p95")
    fig.legend(method_handles, [handle.get_label() for handle in method_handles], loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=6, frameon=False, handlelength=1.6, columnspacing=1.0)
    axes[0].legend(handles=policy_handles + [p95_handle], loc="upper left", bbox_to_anchor=(0.0, 0.99), ncol=3, frameon=False, handletextpad=0.35, columnspacing=0.8)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.16, top=0.82, wspace=0.30)
    fig.text(0.055, 0.055, "All intervals summarize 1,000 held-out snapshots; canonical parity and Hann robustness tables are provided with the source artifacts.", fontsize=7.1, color="0.28")

    stem = "ablation_condT_high_frequency_20260910"
    paths = {extension: output_dir / f"{stem}.{extension}" for extension in ("svg", "pdf", "png")}
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=320, bbox_inches="tight")
    plt.close(fig)
    if "<text" not in paths["svg"].read_text(encoding="utf-8"):
        raise AssertionError("SVG did not retain editable text nodes")
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
        raise AssertionError("One or more figure exports are empty")

    # Keep the compact robustness table beside the figure for reviewers.
    shutil.copy2(input_dir / "U1_hann_robustness_summary.csv", output_dir / "U1_hann_robustness_summary.csv")
    write_contract(output_dir, input_dir, metadata)
    metadata_out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": sha256(Path(__file__).resolve()),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "rows": {
            "summary": int(len(summary)),
            "per_state_u1": int(len(per_state[per_state["field"].eq("U1")])),
            "population_spectra": int(len(spectra)),
            "hann_summary": int(len(hann)),
        },
        "input_hashes": {
            name: sha256(input_dir / name)
            for name in (
                "summary_high_frequency.csv",
                "population_U1_spectra.csv",
                "U1_hann_robustness_summary.csv",
                "analysis_metadata.json",
            )
        },
        "exports": {extension: str(path) for extension, path in paths.items()},
    }
    (output_dir / "plot_metadata.json").write_text(json.dumps(metadata_out, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {output_dir}")


if __name__ == "__main__":
    main()

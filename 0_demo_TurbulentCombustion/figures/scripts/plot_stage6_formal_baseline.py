#!/usr/bin/env python3
"""Analyze and plot the Stage 6 F0/F1 current-architecture comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "_CheckNotes/Stage6_formal_baseline"
F0_RUN = PACKAGE / "runs/F0_frozen_current_DemoN9300_20260821_075633"
F1_RUN = PACKAGE / "runs/F1_more_supervision_DemoN9301_20260821_075633"
EVAL = PACKAGE / "evaluation"
OUT = ROOT / "figures/generated/stage6_formal_baseline"

F0_COLOR = "#484878"
F1_COLOR = "#D07A8F"
NEUTRAL = "#606060"
POSITIVE_WORSE = "#B64342"
NEGATIVE_BETTER = "#3775BA"


def style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_json(path: Path):
    return json.loads(path.read_text())


def export(fig: plt.Figure, base: Path, *, tiff: bool = True) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if tiff:
        fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    svg = base.with_suffix(".svg").read_text(encoding="utf-8")
    assert "<text" in svg, "SVG text was converted to paths"


def paired_fixed_manifest() -> tuple[pd.DataFrame, dict]:
    rows = pd.read_csv(EVAL / "fixed_manifest_best.csv")
    labels = list(rows["checkpoint"].drop_duplicates())
    assert len(labels) == 2
    f0_label = next(label for label in labels if label.startswith("F0_"))
    f1_label = next(label for label in labels if label.startswith("F1_"))
    paired = rows.pivot(
        index=["repeat", "batch_index"], columns="checkpoint", values="loss"
    ).reset_index()
    paired["f0_loss"] = paired[f0_label]
    paired["f1_loss"] = paired[f1_label]
    paired["f1_minus_f0"] = paired["f1_loss"] - paired["f0_loss"]
    layout = (
        paired.groupby("batch_index", as_index=False)[
            ["f0_loss", "f1_loss", "f1_minus_f0"]
        ]
        .mean()
        .rename(columns={"batch_index": "manifest_index"})
    )
    diff = layout["f1_minus_f0"].to_numpy()
    sem = stats.sem(diff)
    ci_low, ci_high = stats.t.interval(
        0.95, df=len(diff) - 1, loc=float(diff.mean()), scale=float(sem)
    )
    test = stats.ttest_1samp(diff, popmean=0.0)
    summary = {
        "layouts": int(len(layout)),
        "rf_draws_per_layout": int(paired["repeat"].nunique()),
        "f0_mean_rf_loss": float(paired["f0_loss"].mean()),
        "f1_mean_rf_loss": float(paired["f1_loss"].mean()),
        "f1_minus_f0_mean": float(diff.mean()),
        "f1_minus_f0_ci95_layout": [float(ci_low), float(ci_high)],
        "f1_minus_f0_percent_of_f0": float(
            100.0 * diff.mean() / paired["f0_loss"].mean()
        ),
        "layout_level_paired_t_pvalue": float(test.pvalue),
    }
    layout.to_csv(OUT / "fixed_manifest_layout_source.csv", index=False)
    return layout, summary


def load_reconstruction() -> tuple[pd.DataFrame, dict]:
    rows = []
    for model, folder in (("F0", "F0_best"), ("F1", "F1_best")):
        for item in load_json(EVAL / f"matched_reconstruction/{folder}/summary.json"):
            for field, value in item["relative_l2"].items():
                rows.append(
                    {
                        "model": model,
                        "nfe": int(item["nfe"]),
                        "field": field,
                        "relative_l2": float(value),
                        "checkpoint_epoch": int(item["checkpoint_epoch"]),
                        "condition_checksum_sha256": item[
                            "condition_checksum_sha256"
                        ],
                    }
                )
    frame = pd.DataFrame(rows)
    checksums = frame["condition_checksum_sha256"].unique()
    assert len(checksums) == 1
    pivot = frame.pivot(index=["nfe", "field"], columns="model", values="relative_l2")
    pivot["F1_minus_F0"] = pivot["F1"] - pivot["F0"]
    source = pivot.reset_index()
    source.to_csv(OUT / "reconstruction_source.csv", index=False)
    by_nfe = (
        frame.groupby(["model", "nfe"], as_index=False)["relative_l2"].mean()
    )
    summary = {
        "condition_checksum_sha256": str(checksums[0]),
        "mean_field_relative_l2": {
            f"{row.model}_nfe{int(row.nfe)}": float(row.relative_l2)
            for row in by_nfe.itertuples()
        },
    }
    return source, summary


def efficiency_summary(f0: pd.DataFrame, f1: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d0 = load_json(F0_RUN / "data_path_diagnostics_summary.json")["cumulative"]["mean"]
    d1 = load_json(F1_RUN / "data_path_diagnostics_summary.json")["cumulative"]["mean"]
    values = {
        "Epoch time": (
            float(f0.loc[f0.epoch >= 2, "train_seconds"].mean()),
            float(f1.loc[f1.epoch >= 2, "train_seconds"].mean()),
        ),
        "Training step": (
            float(d0["total_training_step_ms"]),
            float(d1["total_training_step_ms"]),
        ),
        "Reserved memory": (
            float(d0["gpu_peak_reserved_mb"]),
            float(d1["gpu_peak_reserved_mb"]),
        ),
    }
    rows = []
    for metric, (f0_value, f1_value) in values.items():
        rows.append(
            {
                "metric": metric,
                "F0": f0_value,
                "F1": f1_value,
                "F1_over_F0": f1_value / f0_value,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "efficiency_source.csv", index=False)
    summary = {
        "mean_train_seconds_epoch_2_200": {
            "F0": values["Epoch time"][0],
            "F1": values["Epoch time"][1],
        },
        "total_train_hours": {
            "F0": float(f0["train_seconds"].sum() / 3600.0),
            "F1": float(f1["train_seconds"].sum() / 3600.0),
        },
        "diagnostic_step_ms": {
            "F0": values["Training step"][0],
            "F1": values["Training step"][1],
        },
        "sampled_peak_reserved_mb": {
            "F0": values["Reserved memory"][0],
            "F1": values["Reserved memory"][1],
        },
    }
    return frame, summary


def plot_decision(
    f0: pd.DataFrame,
    f1: pd.DataFrame,
    layouts: pd.DataFrame,
    recon: pd.DataFrame,
    efficiency: pd.DataFrame,
    fixed_summary: dict,
) -> None:
    fig = plt.figure(figsize=(183 / 25.4, 116 / 25.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for frame, label, color in ((f0, "F0: 4k", F0_COLOR), (f1, "F1: 16k/8k", F1_COLOR)):
        smooth = frame["train_loss"].rolling(5, center=True, min_periods=1).mean()
        ax_a.plot(frame["epoch"], smooth, color=color, label=f"{label} train")
        valid = frame["val_loss"].notna()
        ax_a.plot(
            frame.loc[valid, "epoch"],
            frame.loc[valid, "val_loss"],
            color=color,
            linestyle="none",
            marker="o",
            markersize=2.3,
            markerfacecolor="white",
            markeredgewidth=0.7,
            label=f"{label} validation",
        )
        best = frame.loc[valid].loc[frame.loc[valid, "val_loss"].idxmin()]
        ax_a.scatter(best["epoch"], best["val_loss"], color=color, marker="*", s=38, zorder=5)
    ax_a.set(xlabel="Epoch", ylabel="RF loss", xlim=(0, 202), ylim=(0.48, 1.72))
    ax_a.legend(ncol=2, loc="upper right", handlelength=1.6, columnspacing=0.8)
    ax_a.set_title("Matched convergence (one seed)")
    panel_label(ax_a, "a")

    x = np.zeros(len(layouts))
    jitter = np.random.default_rng(7).normal(0.0, 0.035, size=len(layouts))
    delta = layouts["f1_minus_f0"].to_numpy()
    ax_b.scatter(x + jitter, delta, s=10, color=NEUTRAL, alpha=0.55, linewidths=0)
    mean = fixed_summary["f1_minus_f0_mean"]
    lo, hi = fixed_summary["f1_minus_f0_ci95_layout"]
    ax_b.plot([0, 0], [lo, hi], color=POSITIVE_WORSE, lw=2.0)
    ax_b.scatter([0], [mean], color=POSITIVE_WORSE, s=34, marker="D", zorder=4)
    ax_b.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_b.set_xlim(-0.24, 0.24)
    ax_b.set_xticks([0], ["F1 − F0"])
    ax_b.set_ylabel("Paired RF-loss difference")
    ax_b.set_title("Fixed manifest: 64 layouts × 3 RF draws")
    ax_b.text(
        0.98,
        0.97,
        f"mean {mean:+.4f}\n95% CI [{lo:+.4f}, {hi:+.4f}]",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    panel_label(ax_b, "b")

    fields = ["CO", "T", "U_0", "U_1", "p"]
    nfes = [1, 2, 4]
    matrix = np.asarray(
        [
            [
                recon.loc[(recon.nfe == nfe) & (recon.field == field), "F1_minus_F0"].iloc[0]
                for nfe in nfes
            ]
            for field in fields
        ]
    )
    vmax = float(np.max(np.abs(matrix)))
    im = ax_c.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_c.set_xticks(range(len(nfes)), [f"NFE {nfe}" for nfe in nfes])
    ax_c.set_yticks(range(len(fields)), fields)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax_c.text(col, row, f"{matrix[row, col]:+.3f}", ha="center", va="center", fontsize=6)
    ax_c.set_title("Matched reconstruction: F1 − F0 relative L2")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("Positive = F1 worse", fontsize=6.5)
    panel_label(ax_c, "c")

    xpos = np.arange(len(efficiency))
    ratio = efficiency["F1_over_F0"].to_numpy()
    bars = ax_d.bar(xpos, ratio, color=F1_COLOR, edgecolor=F0_COLOR, linewidth=0.8)
    ax_d.axhline(1.0, color=F0_COLOR, lw=1.0, linestyle="--", label="F0 = 1")
    ax_d.set_xticks(xpos, efficiency["metric"], rotation=18, ha="right")
    ax_d.set_ylabel("F1 / F0 cost")
    ax_d.set_ylim(0, max(ratio) * 1.22)
    for bar, value in zip(bars, ratio):
        ax_d.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.2f}×", ha="center", va="bottom")
    ax_d.set_title("Cost of larger supervision")
    panel_label(ax_d, "d")

    fig.suptitle(
        "Larger query supervision does not improve the current decoder",
        fontsize=9,
        fontweight="bold",
        y=1.01,
    )
    export(fig, OUT / "stage6_formal_baseline_decision")


def plot_reconstruction_plate(recon_source: pd.DataFrame) -> None:
    f0_npz = np.load(EVAL / "matched_reconstruction/F0_best/nfe1.npz")
    f1_npz = np.load(EVAL / "matched_reconstruction/F1_best/nfe1.npz")
    assert np.array_equal(f0_npz["coords_raw"], f1_npz["coords_raw"])
    assert np.array_equal(f0_npz["truth_phys"], f1_npz["truth_phys"])
    coords = f0_npz["coords_raw"]
    truth = f0_npz["truth_phys"]
    pred0 = f0_npz["recon_phys"]
    pred1 = f1_npz["recon_phys"]
    assert np.isfinite(truth).all() and np.isfinite(pred0).all() and np.isfinite(pred1).all()
    field_names = list(f0_npz["field_names"].astype(str))
    triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1])

    fig, axes = plt.subplots(3, 5, figsize=(183 / 25.4, 112 / 25.4), sharex=True, sharey=True)
    row_data = (("Truth", truth), ("F0 best", pred0), ("F1 best", pred1))
    for col, field in enumerate(field_names):
        values = np.concatenate([truth[:, col], pred0[:, col], pred1[:, col]])
        vmin, vmax = np.quantile(values, [0.005, 0.995])
        for row, (row_name, data) in enumerate(row_data):
            ax = axes[row, col]
            mesh = ax.tripcolor(
                triangulation,
                data[:, col],
                shading="gouraud",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(field, fontweight="bold")
            if col == 0:
                ax.set_ylabel(row_name, fontweight="bold")
            if row > 0:
                model = "F0" if row == 1 else "F1"
                metric = recon_source.loc[
                    (recon_source.nfe == 1) & (recon_source.field == field), model
                ].iloc[0]
                ax.text(
                    0.98,
                    0.03,
                    f"rel. L2 {metric:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=5.5,
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.2},
                )
        cbar = fig.colorbar(mesh, ax=axes[:, col], orientation="horizontal", fraction=0.035, pad=0.025)
        cbar.ax.tick_params(labelsize=5, length=2)

    axes[0, 0].text(
        -0.28,
        1.08,
        "a",
        transform=axes[0, 0].transAxes,
        fontsize=8,
        fontweight="bold",
    )
    fig.suptitle(
        "Matched best-checkpoint reconstruction (snapshot 0, 256 T sensors, Euler NFE 1)",
        fontsize=8.5,
        fontweight="bold",
        y=1.01,
    )
    export(fig, OUT / "stage6_matched_reconstruction_fields")


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    f0 = pd.read_csv(F0_RUN / "loss_history.csv")
    f1 = pd.read_csv(F1_RUN / "loss_history.csv")
    assert len(f0) == len(f1) == 200
    assert f0["epoch"].tolist() == f1["epoch"].tolist() == list(range(1, 201))
    convergence = pd.concat(
        [f0.assign(model="F0"), f1.assign(model="F1")], ignore_index=True
    )
    convergence.to_csv(OUT / "convergence_source.csv", index=False)

    layouts, fixed = paired_fixed_manifest()
    recon, recon_summary = load_reconstruction()
    efficiency, cost_summary = efficiency_summary(f0, f1)

    valid0 = f0.dropna(subset=["val_loss"])
    valid1 = f1.dropna(subset=["val_loss"])
    convergence_summary = {
        "epochs": 200,
        "training_seeds_per_protocol": 1,
        "F0_best_val": float(valid0["val_loss"].min()),
        "F0_best_epoch": int(valid0.loc[valid0["val_loss"].idxmin(), "epoch"]),
        "F1_best_val": float(valid1["val_loss"].min()),
        "F1_best_epoch": int(valid1.loc[valid1["val_loss"].idxmin(), "epoch"]),
        "F0_final_val": float(valid0.iloc[-1]["val_loss"]),
        "F1_final_val": float(valid1.iloc[-1]["val_loss"]),
        "F0_last20_train_mean": float(f0.tail(20)["train_loss"].mean()),
        "F1_last20_train_mean": float(f1.tail(20)["train_loss"].mean()),
    }
    summary = {
        "decision": "retain_F0",
        "reason": (
            "F1 shows no material accuracy improvement at one seed, is slightly worse on "
            "the controlled fixed manifest and matched best-checkpoint reconstruction, "
            "and costs substantially more."
        ),
        "convergence": convergence_summary,
        "fixed_manifest": fixed,
        "matched_reconstruction": recon_summary,
        "efficiency": cost_summary,
        "limitations": [
            "one training seed per protocol",
            "fixed-manifest RF repeats are technical rather than training replicates",
            "matched field plate uses one validation snapshot",
            "best checkpoints occur at epochs 180 (F0) and 200 (F1)",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_decision(f0, f1, layouts, recon, efficiency, fixed)
    plot_reconstruction_plate(recon)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

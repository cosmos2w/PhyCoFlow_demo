#!/usr/bin/env python3
"""Build the source-backed GL_rbf_CQ RC1 documentation figure set.

This is a reporting script only. It reads committed benchmark/evaluation tables
and the deterministic three-snapshot reconstruction package; it does not train
or mutate a model checkpoint.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "figures/generated"
S7 = ROOT / "_CheckNotes/Stage7_smart_cq/evaluation_1000"
RC_EVAL = ROOT / "_CheckNotes/GL_rbf_CQ_rc1_evaluation/matched_reconstruction"

COLORS = {
    "GL_rbf_ENH": "#555555",
    "F0-e1000": "#555555",
    "CQ-LR-128": "#4C78A8",
    "CQ-LR-128-e1000": "#4C78A8",
    "CQ-LR-256": "#2A9D8F",
    "CQ-LR-256-best-e840-partial": "#2A9D8F",
    "GL_rbf_CQ": "#8B5FBF",
    "S7-B-e1000": "#8B5FBF",
}
DISPLAY = {
    "GL_rbf_ENH": "GL_rbf_ENH (F0)",
    "F0-e1000": "GL_rbf_ENH (F0)",
    "CQ-LR-128": "CQ-LR-128",
    "CQ-LR-128-e1000": "CQ-LR-128",
    "CQ-LR-256": "CQ-LR-256†",
    "CQ-LR-256-best-e840-partial": "CQ-LR-256†",
    "GL_rbf_CQ": "GL_rbf_CQ",
    "S7-B-e1000": "GL_rbf_CQ",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, weight="bold", fontsize=9)


def export(fig: plt.Figure, directory: str, stem: str) -> None:
    out = FIG_ROOT / directory
    out.mkdir(parents=True, exist_ok=True)
    base = out / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    assert "<text" in base.with_suffix(".svg").read_text(), "SVG text not editable"


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    fontsize: float = 6.5,
    edge: str | None = None,
    linestyle: str = "-",
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=color,
        edgecolor=edge or color,
        linewidth=0.9,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#555555") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=color))


def architecture_figure() -> None:
    stages = [
        (1, "Matched baseline\n+ instrumentation", "execution", "#E8E8E8"),
        (2, "Selected-only\ndata path", "execution", "#DCEAF5"),
        (3, "Scaling diagnosis\n4k→65k", "execution", "#DCEAF5"),
        (4, "Cached-streamed\nreconstruction", "execution", "#DCEAF5"),
        (5, "Query\nmicrobatching", "execution", "#DCEAF5"),
        (6, "Compact CQ-LR\npersistent Top-K", "architecture", "#C9E7DF"),
        (7, "Smart condition\nEMA/FiLM/raw RBF", "architecture", "#DDCFF0"),
    ]
    out = FIG_ROOT / "gl_rbf_cq_rc1_architecture"
    pd.DataFrame(stages, columns=["stage", "change", "change_class", "color"]).to_csv(out / "architecture_source.csv", index=False)

    fig = plt.figure(figsize=(183 / 25.4, 104 / 25.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.62, 1.38], hspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    for ax in (ax_a, ax_b):
        ax.set_axis_off()

    x_positions = np.linspace(0.015, 0.885, len(stages))
    for idx, ((stage, change, _, color), x) in enumerate(zip(stages, x_positions)):
        rounded_box(ax_a, (x, 0.22), 0.10, 0.54, f"Stage {stage}\n{change}", color, fontsize=5.2)
        if idx < len(stages) - 1:
            arrow(ax_a, (x + 0.102, 0.49), (x_positions[idx + 1] - 0.005, 0.49))
    ax_a.text(0.015, 0.04, "Execution-only changes", color="#4C78A8", fontsize=6.2)
    ax_a.text(0.70, 0.04, "Model/training changes", color="#6D3FA0", fontsize=6.2)
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    panel_label(ax_a, "a", x=-0.02, y=1.02)

    boxes = [
        (0.01, 0.57, 0.13, "Sparse sensors\n(x, field, value)", "#E9EEF3"),
        (0.18, 0.57, 0.16, "Sensor tokens\nFourier + field ID", "#DCEAF5"),
        (0.39, 0.57, 0.17, "Latent core\n128 tokens × 256-D", "#D8E9E5"),
        (0.62, 0.57, 0.15, "Global readout\n128-D low rank", "#D8E9E5"),
        (0.82, 0.57, 0.16, "Compact head\n128-D additive", "#DDCFF0"),
        (0.18, 0.12, 0.16, "Persistent geometry\nTop-K=32 + d²", "#F2E6CA"),
        (0.39, 0.12, 0.17, "Local RBF stream\nlearned + raw value/support", "#F2E6CA"),
        (0.62, 0.12, 0.15, "Point stream\n128-D + zero-init FiLM", "#DDCFF0"),
        (0.82, 0.12, 0.16, "5 RF velocity fields\n+ compact GLRES branch", "#DDCFF0"),
    ]
    for x, y, w, label, color in boxes:
        rounded_box(ax_b, (x, y), w, 0.25, label, color, fontsize=6.1)
    for s, e in [
        ((0.14, 0.695), (0.18, 0.695)), ((0.34, 0.695), (0.39, 0.695)),
        ((0.56, 0.695), (0.62, 0.695)), ((0.77, 0.695), (0.82, 0.695)),
        ((0.095, 0.57), (0.22, 0.37)), ((0.34, 0.245), (0.39, 0.245)),
        ((0.56, 0.245), (0.62, 0.245)), ((0.77, 0.245), (0.82, 0.245)),
        ((0.895, 0.57), (0.895, 0.37)),
    ]:
        arrow(ax_b, s, e)
    ax_b.plot([0.155, 0.155], [0.02, 0.46], color="#C48720", linestyle="--", linewidth=0.9)
    ax_b.text(0.158, 0.03, "cache boundary: no post-build KNN", color="#9A6515", fontsize=5.9, rotation=90, va="bottom")
    ax_b.text(0.01, 0.96, "Frozen GL_rbf_CQ RC1", weight="bold", fontsize=7.8, va="top")
    ax_b.text(0.01, 0.88, "Condition-static caches remain time independent; FiLM acts only on point_q.", fontsize=6.2, color="#444444")
    ax_b.text(0.01, 0.02, "Rejected at cost gate: 192/224-D structured-concat CQ (insufficient F0 speed/memory margin).", fontsize=6.0, color="#A04A3A")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    panel_label(ax_b, "b", x=-0.02, y=1.02)
    fig.suptitle("GL_rbf_CQ: execution-first evolution to a compact, condition-aware RF model", fontsize=9.4, weight="bold")
    export(fig, "gl_rbf_cq_rc1_architecture", "gl_rbf_cq_rc1_architecture")


def convergence_figure() -> None:
    conv = pd.read_csv(S7 / "convergence.csv")
    comp = pd.read_csv(S7 / "final_comparison.csv").set_index("candidate")
    out = FIG_ROOT / "gl_rbf_cq_rc1_convergence"
    conv.to_csv(out / "convergence_source.csv", index=False)
    comp.reset_index().to_csv(out / "endpoint_source.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 92 / 25.4), gridspec_kw={"width_ratios": [1.45, 0.88, 0.87]})
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.78, wspace=0.46)
    ax_a, ax_b, ax_c = axes
    mapping = {
        "F0": "F0-e1000", "CQ-LR-128": "CQ-LR-128-e1000",
        "CQ-LR-256 (partial)": "CQ-LR-256-best-e840-partial", "S7-B": "S7-B-e1000",
    }
    for raw, canonical in mapping.items():
        rows = conv[conv.candidate == raw].sort_values("epoch")
        partial = "partial" in raw
        ax_a.plot(rows.epoch, rows.mean_rf_loss, marker="o", markersize=2.8,
                  markerfacecolor="white" if partial else COLORS[canonical],
                  markeredgecolor=COLORS[canonical], linestyle="--" if partial else "-",
                  color=COLORS[canonical], label=DISPLAY[canonical])
    ax_a.axhline(float(comp.loc["F0-e1000", "fixed_manifest_rf_mean"]), color="#999999", linewidth=0.7, linestyle=":")
    ax_a.set_yscale("log")
    ax_a.set(xlabel="Epoch", ylabel="Fixed-manifest RF loss", xlim=(-20, 1020), ylim=(0.235, 2.2))
    ax_a.set_yticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1.0", "2.0"])
    ax_a.legend(ncols=2, loc="upper right")
    ax_a.set_title("Crosses the F0 endpoint by epoch 400", loc="left", weight="bold")

    order = ["F0-e1000", "CQ-LR-128-e1000", "CQ-LR-256-best-e840-partial", "S7-B-e1000"]
    vals = [float(comp.loc[k, "fixed_manifest_rf_mean"]) for k in order]
    cis = [[float(comp.loc[k, "fixed_manifest_rf_ci95_low"]), float(comp.loc[k, "fixed_manifest_rf_ci95_high"])] for k in order]
    y = np.arange(len(order))
    for ypos, key, value, ci in zip(y, order, vals, cis):
        ax_b.errorbar(value, ypos, xerr=[[value-ci[0]], [ci[1]-value]], fmt="o",
                      color=COLORS[key], markerfacecolor="white" if "partial" in key else COLORS[key], capsize=2.3)
    endpoint_labels = ["F0", "CQ-LR-128", "CQ-LR-256†", "GL_rbf_CQ"]
    ax_b.set_yticks(y, endpoint_labels)
    ax_b.invert_yaxis()
    ax_b.set(xlabel="RF loss (mean ± 95% CI)", xlim=(0.225, 0.395))
    ax_b.set_title("Matched endpoints (n=192)", loc="left", weight="bold")

    f0 = float(comp.loc["F0-e1000", "fixed_manifest_rf_mean"])
    compared = order[1:]
    gains = [-100 * float(comp.loc[k, "paired_difference_vs_f0_e1000_mean"]) / f0 for k in compared]
    bars = ax_c.bar(np.arange(3), gains, color=[COLORS[k] for k in compared], edgecolor=[COLORS[k] for k in compared], linewidth=1.0, width=0.68)
    bars[1].set_facecolor("white")
    ax_c.axhline(0, color="#777777", linewidth=0.7)
    ax_c.set_xticks(np.arange(3), [DISPLAY[k] for k in compared], rotation=22, ha="right")
    ax_c.set_ylabel("Paired improvement vs F0 (%)")
    ax_c.set_ylim(-13, 24)
    ax_c.set_title("Paired change versus F0", loc="left", weight="bold")
    for x, value in enumerate(gains):
        ax_c.text(x, value + (0.8 if value >= 0 else -0.8), f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=6.2)
    for label, ax in zip("abc", axes):
        panel_label(ax, label, x=-0.13, y=1.10)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Latent-256 condition capacity restores compact-query RF quality", fontsize=9.4, weight="bold")
    export(fig, "gl_rbf_cq_rc1_convergence", "gl_rbf_cq_rc1_convergence")


def aggregate_reconstruction() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    protocol: list[dict[str, object]] = []
    for snapshot in (0, 1, 2):
        summary = json.loads((RC_EVAL / f"snapshot_{snapshot:04d}/summary.json").read_text())
        protocol.append(summary["protocol"])
        for candidate, result in summary["candidates"].items():
            for nfe, metrics in result["nfe"].items():
                for field, value in metrics["field_relative_l2"].items():
                    rows.append({"snapshot": snapshot, "candidate": candidate, "nfe": int(nfe), "field": field, "relative_l2": value, "weights": result["weights"], "epoch": result["epoch"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(RC_EVAL / "three_snapshot_source.csv", index=False)
    summary = {
        "protocols": protocol,
        "n_snapshots": 3,
        "aggregation": "arithmetic mean over field-relative L2 values; points show each snapshot",
        "nfe4_candidate_mean": frame[frame.nfe == 4].groupby("candidate").relative_l2.mean().to_dict(),
        "nfe4_u1_mean": frame[(frame.nfe == 4) & (frame.field == "U_1")].groupby("candidate").relative_l2.mean().to_dict(),
    }
    (RC_EVAL / "three_snapshot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    out = FIG_ROOT / "gl_rbf_cq_rc1_reconstruction"
    frame.to_csv(out / "reconstruction_source.csv", index=False)
    return frame


def crop_field_plate(path: Path, section: str) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"))
    h, w = image.shape[:2]
    ranges = {"truth": (0.105, 0.355), "reconstruction": (0.435, 0.675)}
    y0, y1 = ranges[section]
    return image[int(h*y0):int(h*y1), int(w*0.01):int(w*0.86)]


def reconstruction_figure() -> None:
    frame = aggregate_reconstruction()
    candidates = ["GL_rbf_ENH", "CQ-LR-128", "CQ-LR-256", "GL_rbf_CQ"]
    fig = plt.figure(figsize=(183 / 25.4, 122 / 25.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.48, 1.0], height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.30)
    plate_gs = gs[:, 0].subgridspec(5, 1, hspace=0.08)
    plate_axes = [fig.add_subplot(plate_gs[i, 0]) for i in range(5)]
    base = RC_EVAL / "snapshot_0000"
    truth_path = base / "GL_rbf_CQ/euler_nfe4_field_U_1.png"
    plate_axes[0].imshow(crop_field_plate(truth_path, "truth"))
    plate_axes[0].text(0.01, 0.90, "Truth: U₁", transform=plate_axes[0].transAxes, color="black", fontsize=6.5,
                       bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.2})
    for ax, candidate in zip(plate_axes[1:], candidates):
        path = base / candidate / "euler_nfe4_field_U_1.png"
        ax.imshow(crop_field_plate(path, "reconstruction"))
        metric = frame[(frame.snapshot == 0) & (frame.candidate == candidate) & (frame.nfe == 4) & (frame.field == "U_1")].relative_l2.iloc[0]
        ax.text(0.01, 0.90, f"{DISPLAY[candidate]}  |  rel. L2={metric:.3f}", transform=ax.transAxes, color="black", fontsize=6.2,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.2})
    for ax in plate_axes:
        ax.set_axis_off()
    panel_label(plate_axes[0], "a", x=-0.06, y=1.12)
    plate_axes[0].set_title("Representative matched snapshot, Euler NFE-4", loc="left", weight="bold", pad=7)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    for ax, subset, title in [
        (ax_b, frame[frame.nfe == 4], "Five-field mean"),
        (ax_c, frame[(frame.nfe == 4) & (frame.field == "U_1")], "Worst field: U₁"),
    ]:
        if title == "Five-field mean":
            points = subset.groupby(["candidate", "snapshot"], as_index=False).relative_l2.mean()
        else:
            points = subset
        for i, candidate in enumerate(candidates):
            vals = points[points.candidate == candidate].relative_l2.to_numpy()
            ax.scatter(np.full(len(vals), i), vals, s=17, facecolor="white", edgecolor=COLORS[candidate], linewidth=0.9, zorder=3)
            ax.plot([i-0.22, i+0.22], [vals.mean(), vals.mean()], color=COLORS[candidate], linewidth=2.2)
        ax.set_xticks(np.arange(4), [DISPLAY[k] for k in candidates], rotation=21, ha="right")
        ax.set_ylabel("Field-relative L2")
        ax.set_title(title + " (n=3 snapshots)", loc="left", weight="bold")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
        ax.set_axisbelow(True)
    panel_label(ax_b, "b", x=-0.15, y=1.08)
    panel_label(ax_c, "c", x=-0.15, y=1.08)
    ax_b.set_ylim(0.20, 0.34)
    ax_c.set_ylim(0.40, 0.75)
    ax_c.text(0.02, 0.04, "Points: snapshots 0–2\nBars: arithmetic mean", transform=ax_c.transAxes, fontsize=5.9, color="#555555")
    fig.suptitle("Matched reconstruction confirms the quality recovery and the remaining U₁ limitation", fontsize=9.4, weight="bold")
    export(fig, "gl_rbf_cq_rc1_reconstruction", "gl_rbf_cq_rc1_reconstruction")


def pareto_figure() -> None:
    comp = pd.read_csv(S7 / "final_comparison.csv").set_index("candidate")
    order = ["F0-e1000", "CQ-LR-128-e1000", "S7-B-e1000"]
    out = FIG_ROOT / "gl_rbf_cq_rc1_pareto"
    comp.loc[order].reset_index().to_csv(out / "pareto_source.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 96 / 25.4))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.81, wspace=0.46)
    ax_a, ax_b, ax_c = axes
    offsets = {"F0-e1000": (-18, 8), "CQ-LR-128-e1000": (-25, -26), "S7-B-e1000": (-32, 8)}
    for key in order:
        row = comp.loc[key]
        for ax, xcol in ((ax_a, "train_step_ms"), (ax_b, "train_peak_allocated_mb")):
            x = float(row[xcol]) / (1024 if xcol.endswith("mb") else 1)
            y = float(row.fixed_manifest_rf_mean)
            ax.scatter(x, y, s=62, color=COLORS[key], edgecolor="white", linewidth=0.7, zorder=3)
            ax.annotate(DISPLAY[key], (x, y), xytext=offsets[key], textcoords="offset points", fontsize=6.1, color=COLORS[key])
    ax_a.set(xlabel="B128/Q4096 training step (ms)", ylabel="Fixed-manifest RF loss", xlim=(370, 570), ylim=(0.245, 0.375))
    ax_a.set_title("Quality versus step time", loc="left", weight="bold")
    ax_b.set(xlabel="Peak allocated memory (GiB)", ylabel="Fixed-manifest RF loss", xlim=(18.5, 28), ylim=(0.245, 0.375))
    ax_b.set_title("Quality versus memory", loc="left", weight="bold")
    x = np.arange(3)
    latencies = [float(comp.loc[k, "persistent_1m_nfe4_s"]) for k in order]
    bars = ax_c.bar(x, latencies, color=[COLORS[k] for k in order], width=0.66)
    ax_c.set_xticks(x, [DISPLAY[k] for k in order], rotation=20, ha="right")
    ax_c.set_ylabel("Persistent 1M-query NFE-4 (s)")
    ax_c.set_ylim(0, 0.50)
    ax_c.set_title("Persistent inference", loc="left", weight="bold")
    for bar, key, value in zip(bars, order, latencies):
        params = float(comp.loc[key, "total_parameters"]) / 1e6
        ax_c.text(bar.get_x()+bar.get_width()/2, value+0.015, f"{value:.3f} s\n{params:.2f} M", ha="center", va="bottom", fontsize=5.9)
    for label, ax in zip("abc", axes):
        panel_label(ax, label)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
        ax.set_axisbelow(True)
    ax_a.text(0.03, 0.04, "Lower left is better", transform=ax_a.transAxes, fontsize=5.9, color="#555555")
    ax_b.text(0.03, 0.04, "Lower left is better", transform=ax_b.transAxes, fontsize=5.9, color="#555555")
    fig.suptitle("GL_rbf_CQ is the balanced quality–throughput Pareto recommendation", fontsize=9.4, weight="bold")
    export(fig, "gl_rbf_cq_rc1_pareto", "gl_rbf_cq_rc1_pareto")


def execution_figure() -> None:
    stage2_full = pd.read_csv(ROOT / "_CheckNotes/Stage2_data_path/optimized_fullnorm.csv")
    stage2_sel = pd.read_csv(ROOT / "_CheckNotes/Stage2_data_path/optimized_selectednorm.csv")
    stage4 = pd.read_csv(ROOT / "_CheckNotes/Stage4_reconstruction/reconstruction_scaling.csv")
    stage5 = pd.read_csv(ROOT / "_CheckNotes/Stage5_query_microbatch/query_microbatch_scaling.csv")
    out = FIG_ROOT / "gl_rbf_cq_rc1_execution"
    stage2_full.assign(normalization="full").to_csv(out / "stage2_full_source.csv", index=False)
    stage2_sel.assign(normalization="selected").to_csv(out / "stage2_selected_source.csv", index=False)
    stage4.to_csv(out / "stage4_source.csv", index=False)
    stage5.to_csv(out / "stage5_source.csv", index=False)

    fig = plt.figure(figsize=(183 / 25.4, 96 / 25.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.48, 1.52], hspace=0.38, wspace=0.35)
    ax_top = fig.add_subplot(gs[0, :])
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[1, i]) for i in range(3)]
    ax_top.set_axis_off()
    timeline = [
        ("1", "Matched order\n+ diagnostics", "equivalence"),
        ("2", "Selected-only\nnormalization", "4k: −19.6% pre-model"),
        ("3", "Scaling\ndiagnosis", "query work dominates"),
        ("4", "Cached-streamed\nreconstruction", "250k: 6.70× faster"),
        ("5", "Query\nmicrobatching", "65k: −89.3% memory"),
    ]
    xs = np.linspace(0.02, 0.82, 5)
    for i, ((stage, title, note), xpos) in enumerate(zip(timeline, xs)):
        rounded_box(ax_top, (xpos, 0.16), 0.16, 0.66, f"Stage {stage}\n{title}\n{note}", "#DCEAF5", fontsize=5.8)
        if i < 4:
            arrow(ax_top, (xpos+0.162, 0.49), (xs[i+1]-0.006, 0.49))
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)

    queries = [4096, 16384]
    full = [float(stage2_full.loc[stage2_full.N_query == q, "pre_model_latency_ms"].iloc[0]) for q in queries]
    selected = [float(stage2_sel.loc[stage2_sel.N_query == q, "pre_model_latency_ms"].iloc[0]) for q in queries]
    x = np.arange(2)
    ax_a.bar(x-0.18, full, width=0.36, color="#B8B8B8", label="full normalize")
    ax_a.bar(x+0.18, selected, width=0.36, color="#4C78A8", label="selected normalize")
    ax_a.set_xticks(x, ["4k", "16k"])
    ax_a.set(xlabel="Queries", ylabel="Pre-model latency (ms)")
    ax_a.legend()
    ax_a.set_title("Stage 2 data-path crossover", loc="left", weight="bold")

    legacy = stage4[stage4.execution_mode == "legacy_full"]
    cached = stage4[stage4.execution_mode == "cached_streamed"]
    ax_b.plot(legacy.N_query, legacy.seconds_per_million_points_per_nfe, "o--", color="#777777", label="legacy full")
    ax_b.plot(cached.N_query, cached.seconds_per_million_points_per_nfe, "o-", color="#2A9D8F", label="cached streamed")
    ax_b.set_xscale("log")
    ax_b.set_xticks([40300, 250000, 1000000], ["40k", "250k", "1M"])
    ax_b.set(xlabel="Reconstruction points", ylabel="s / 1M points / NFE")
    ax_b.set_title("Stage 4 makes 1M feasible", loc="left", weight="bold")
    ax_b.legend()

    mono = stage5[stage5.execution == "monolithic"].set_index("N_query_effective")
    micro = stage5[stage5.query_microbatch_size == 4096].set_index("N_query_effective")
    qs = [16384, 65536]
    for i, q in enumerate(qs):
        ax_c.scatter(mono.loc[q, "step_ms"], mono.loc[q, "gpu_peak_allocated_mb"], s=45, color="#777777", marker="o")
        ax_c.scatter(micro.loc[q, "step_ms"], micro.loc[q, "gpu_peak_allocated_mb"], s=45, color="#8B5FBF", marker="s")
        ax_c.annotate(f"{q//1024}k", (micro.loc[q, "step_ms"], micro.loc[q, "gpu_peak_allocated_mb"]), xytext=(4, 4), textcoords="offset points", fontsize=6)
    ax_c.set(xlabel="Training step (ms)", ylabel="Peak allocated memory (MB)")
    ax_c.set_title("Stage 5 bounds activation memory", loc="left", weight="bold")
    ax_c.text(0.02, 0.95, "● monolithic   ■ microbatch 4k", transform=ax_c.transAxes, va="top", fontsize=5.9)
    for label, ax in zip("abc", (ax_a, ax_b, ax_c)):
        panel_label(ax, label)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Stages 1–5 separate execution scaling from scientific architecture changes", fontsize=9.4, weight="bold")
    export(fig, "gl_rbf_cq_rc1_execution", "gl_rbf_cq_rc1_execution")


def main() -> None:
    style()
    architecture_figure()
    convergence_figure()
    reconstruction_figure()
    pareto_figure()
    execution_figure()


if __name__ == "__main__":
    main()

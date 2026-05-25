#!/usr/bin/env python3
"""Build the turbulent-combustion conditioning ablation paper figure.

The script uses existing trained/evaluated artifacts only. It does not load
model checkpoints or run inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phycoflow_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")
FIELD_LABELS = {"CH4": "CH4", "CO": "CO", "T": "T", "U_1": "U1", "p": "p"}
FIELD_INDEX_TO_NAME = {0: "CH4", 1: "CO", 2: "T", 3: "U_1", 4: "p"}

FIGURE_NAME = "turbulent_combustion_conditioning_ablation"
DEFAULT_OUTPUT_DIR = Path("figures/generated") / FIGURE_NAME

MODEL_COLORS = {
    "N0": "#6C757D",
    "N15": "#0072B2",
    "N17": "#D55E00",
    "N18": "#009E73",
}

FIELD_COLORS = {
    "CH4": "#5DA5DA",
    "CO": "#F17CB0",
    "T": "#F15854",
    "U_1": "#60BD68",
    "p": "#B2912F",
}


@dataclass(frozen=True)
class DemoSpec:
    key: str
    demo_num: int
    timestamp: str
    label: str
    short_label: str
    model_dir: Path
    recon_dir: Path
    loss_csv: Path
    config_json: Path
    config_yaml: Path
    final_epoch: int


def project_path(*parts: str) -> Path:
    return Path(*parts)


def build_specs(root: Path) -> list[DemoSpec]:
    demo_root = root / "0_demo_TurbulentCombustion"
    return [
        DemoSpec(
            key="N0",
            demo_num=0,
            timestamp="20260519_164357",
            label="N0 FNO baseline",
            short_label="N0",
            model_dir=demo_root / "Save_TrainedModel/ffm_tc_pointcloud_DemoN0_20260519_164357",
            recon_dir=demo_root / "Save_reconstruction_files/ffm_tc_pointcloud/demo_N0_20260519_164357",
            loss_csv=demo_root / "Save_loss_csv/Loss_DemoN0_20260519_164357/losses.csv",
            config_json=demo_root / "Save_TrainedModel/ffm_tc_pointcloud_DemoN0_20260519_164357/args.json",
            config_yaml=demo_root / "Save_config/pointcloud_ffm/config_pointcloud_ffm_DemoN0_20260519_164357.yaml",
            final_epoch=3500,
        ),
        DemoSpec(
            key="N15",
            demo_num=15,
            timestamp="20260513_083830",
            label="N15 GL-rbf T+U1",
            short_label="N15",
            model_dir=demo_root / "Save_TrainedModel/ffm_tc_pointcloud_DemoN15_20260513_083830",
            recon_dir=demo_root / "Save_reconstruction_files/ffm_tc_pointcloud/demo_N15_20260513_083830",
            loss_csv=demo_root / "Save_loss_csv/Loss_DemoN15_20260513_083830/losses.csv",
            config_json=demo_root / "Save_TrainedModel/ffm_tc_pointcloud_DemoN15_20260513_083830/args.json",
            config_yaml=demo_root / "Save_config/pointcloud_ffm/config_pointcloud_ffm_DemoN15_20260513_083830.yaml",
            final_epoch=9500,
        ),
        DemoSpec(
            key="N17",
            demo_num=17,
            timestamp="20260510_082814",
            label="N17 GL-rbf T only",
            short_label="N17",
            model_dir=demo_root / "Save_TrainedModel/ffm_tc_pointcloud_DemoN17_20260510_082814",
            recon_dir=demo_root / "Save_reconstruction_files/ffm_tc_pointcloud/demo_N17_20260510_082814",
            loss_csv=demo_root / "Save_loss_csv/Loss_DemoN17_20260510_082814/losses.csv",
            config_json=demo_root / "Save_TrainedModel/ffm_tc_pointcloud_DemoN17_20260510_082814/args.json",
            config_yaml=demo_root / "Save_config/pointcloud_ffm/config_pointcloud_ffm_DemoN17_20260510_082814.yaml",
            final_epoch=8000,
        ),
        DemoSpec(
            key="N18",
            demo_num=18,
            timestamp="20260511_113906",
            label="N18 GL-rbf CO+T+U1+p",
            short_label="N18",
            model_dir=demo_root / "Save_TrainedModel/ffm_tc_pointcloud_DemoN18_20260511_113906",
            recon_dir=demo_root / "Save_reconstruction_files/ffm_tc_pointcloud/demo_N18_20260511_113906",
            loss_csv=demo_root / "Save_loss_csv/Loss_DemoN18_20260511_113906/losses.csv",
            config_json=demo_root / "Save_TrainedModel/ffm_tc_pointcloud_DemoN18_20260511_113906/args.json",
            config_yaml=demo_root / "Save_config/pointcloud_ffm/config_pointcloud_ffm_DemoN18_20260511_113906.yaml",
            final_epoch=9000,
        ),
    ]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> dict:
    with require_file(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_loss_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with require_file(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {"epoch": float(row["epoch"])}
            for key in ("train_loss", "val_loss"):
                value = row.get(key, "")
                parsed[key] = float(value) if value not in ("", None) else np.nan
            rows.append(parsed)
    return rows


def metric_path(spec: DemoSpec, epoch: int, nfe: int) -> Path:
    return spec.recon_dir / f"Epoch_{epoch}" / f"euler_nfe{nfe}_metrics.json"


def field_image_path(spec: DemoSpec, epoch: int, nfe: int, field: str) -> Path:
    return spec.recon_dir / f"Epoch_{epoch}" / f"euler_nfe{nfe}_field_{field}.png"


def numeric_field_metrics(payload: dict) -> dict[str, float]:
    raw = payload["metrics"]
    return {
        field: float(raw[field])
        for field in FIELD_NAMES
        if field in raw and math.isfinite(float(raw[field]))
    }


def list_eval_epochs(spec: DemoSpec, nfe: int = 8) -> list[int]:
    epochs = []
    for epoch_dir in spec.recon_dir.glob("Epoch_*"):
        if not epoch_dir.is_dir():
            continue
        try:
            epoch = int(epoch_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if (epoch_dir / f"euler_nfe{nfe}_metrics.json").exists():
            epochs.append(epoch)
    return sorted(epochs)


def mean_l2(metrics: dict[str, float]) -> float:
    return float(np.mean([metrics[field] for field in FIELD_NAMES if field in metrics]))


def field_list_label(indices: Iterable[int]) -> str:
    return "+".join(FIELD_INDEX_TO_NAME.get(int(i), f"f{i}") for i in indices)


def crop_reconstruction_strip(path: Path, strip: str) -> Image.Image:
    img = Image.open(require_file(path)).convert("RGB")
    w, h = img.size
    x0, x1 = int(0.01 * w), int(0.90 * w)
    if strip == "truth":
        y0, y1 = int(0.105 * h), int(0.345 * h)
    elif strip == "recon":
        y0, y1 = int(0.405 * h), int(0.645 * h)
    elif strip == "error":
        y0, y1 = int(0.720 * h), int(0.970 * h)
    else:
        raise ValueError(f"Unknown strip: {strip}")
    return img.crop((x0, y0, x1, y1))


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.10,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )


def plot_config_table(ax, specs: list[DemoSpec], configs: dict[str, dict], final_metrics: dict[str, dict]) -> None:
    ax.axis("off")
    add_panel_label(ax, "a")
    ax.set_title("Model family and sparse-conditioning design", loc="left", pad=2, fontsize=7.6, fontweight="bold")

    headers = ["demo", "backbone", "conditioned fields", "sensors / field", "Fourier PE", "final epoch", "mean rel. L2"]
    rows = []
    for spec in specs:
        cfg = configs[spec.key]
        obs = cfg.get("vis_n_obs_list") or cfg.get("n_obs_max_list") or []
        rows.append(
            [
                spec.short_label,
                cfg.get("backbone", ""),
                field_list_label(cfg.get("vis_cond_fields") or cfg.get("cond_fields") or []),
                "+".join(str(v) for v in obs),
                "yes" if cfg.get("USE_FOURIER_PE", False) else "no",
                str(spec.final_epoch),
                f"{mean_l2(final_metrics[spec.key]):.3f}",
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.78],
        colWidths=[0.08, 0.13, 0.25, 0.18, 0.12, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.0)
    table.scale(1.0, 1.18)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#F1F3F5")
            cell.set_text_props(weight="bold", color="#212529")
        else:
            key = specs[row - 1].key
            if col == 0:
                cell.set_facecolor(MODEL_COLORS[key])
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#FFFFFF")


def plot_training_trend(ax, specs: list[DemoSpec]) -> None:
    add_panel_label(ax, "b")
    ax.set_title("Checkpoint-time reconstruction trend", loc="left", pad=2, fontsize=7.6, fontweight="bold")
    for spec in specs:
        xs = []
        ys = []
        for epoch in list_eval_epochs(spec, nfe=8):
            payload = load_json(metric_path(spec, epoch, 8))
            xs.append(epoch)
            ys.append(mean_l2(numeric_field_metrics(payload)))
        ax.plot(
            xs,
            ys,
            color=MODEL_COLORS[spec.key],
            lw=1.3,
            marker="o",
            ms=2.2,
            markevery=max(1, len(xs) // 7),
            label=spec.short_label,
        )
    ax.set_xlabel("training epoch")
    ax.set_ylabel("mean relative L2, snapshot 0")
    ax.set_yscale("log")
    ax.grid(True, which="major", color="#E9ECEF", linewidth=0.5)
    ax.legend(ncol=2, loc="upper right", fontsize=6.1, handlelength=1.2, columnspacing=0.8)


def plot_final_bars(ax, specs: list[DemoSpec], final_metrics: dict[str, dict]) -> None:
    add_panel_label(ax, "c")
    ax.set_title("Final field-wise reconstruction error", loc="left", pad=2, fontsize=7.6, fontweight="bold")
    x = np.arange(len(FIELD_NAMES))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(specs))
    for offset, spec in zip(offsets, specs):
        values = [final_metrics[spec.key][field] for field in FIELD_NAMES]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=MODEL_COLORS[spec.key],
            edgecolor="white",
            linewidth=0.25,
            label=spec.short_label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([FIELD_LABELS[f] for f in FIELD_NAMES])
    ax.set_ylabel("relative L2, snapshot 0")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", color="#E9ECEF", linewidth=0.5)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=5.8, handlelength=1.0, columnspacing=0.65)


def plot_nfe_sensitivity(ax, specs: list[DemoSpec]) -> None:
    add_panel_label(ax, "d")
    ax.set_title("Final sampling-step sensitivity", loc="left", pad=2, fontsize=7.6, fontweight="bold")
    nfe_values = [2, 4, 8]
    for spec in specs:
        values = []
        for nfe in nfe_values:
            payload = load_json(metric_path(spec, spec.final_epoch, nfe))
            values.append(mean_l2(numeric_field_metrics(payload)))
        ax.plot(nfe_values, values, color=MODEL_COLORS[spec.key], lw=1.3, marker="o", ms=3.0, label=spec.short_label)
    ax.set_xticks(nfe_values)
    ax.set_xlabel("NFE")
    ax.set_ylabel("mean relative L2")
    ax.set_ylim(0.085, 0.275)
    ax.set_yticks([0.10, 0.15, 0.20, 0.25])
    ax.grid(True, which="major", color="#E9ECEF", linewidth=0.5)
    ax.legend(ncol=2, loc="upper right", fontsize=5.8, handlelength=1.1, columnspacing=0.7)


def plot_multifield_reconstruction_plate(fig, outer_spec, specs: list[DemoSpec]) -> None:
    gs = gridspec.GridSpecFromSubplotSpec(
        6,
        5,
        subplot_spec=outer_spec,
        hspace=0.10,
        wspace=0.035,
        height_ratios=[0.22, 1, 1, 1, 1, 1],
        width_ratios=[1, 1, 1, 1, 1],
    )
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    add_panel_label(title_ax, "e")
    title_ax.text(
        0.0,
        0.72,
        "Multi-field reconstruction comparison",
        ha="left",
        va="center",
        fontsize=7.6,
        fontweight="bold",
    )
    title_ax.text(
        0.99,
        0.72,
        "existing PNG evaluation outputs; NFE=8",
        ha="right",
        va="center",
        fontsize=5.9,
        color="#6C757D",
    )

    col_titles = ["ground truth", *[spec.short_label for spec in specs]]
    title_colors = ["#343A40", *[MODEL_COLORS[spec.key] for spec in specs]]

    for row_idx, field in enumerate(FIELD_NAMES, start=1):
        truth_source = field_image_path(specs[-1], specs[-1].final_epoch, 8, field)
        row_images = [crop_reconstruction_strip(truth_source, "truth")]
        row_images.extend(
            crop_reconstruction_strip(field_image_path(spec, spec.final_epoch, 8, field), "recon")
            for spec in specs
        )

        for col_idx, image in enumerate(row_images):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(image, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.38)
                spine.set_edgecolor("#ADB5BD")
            if row_idx == 1:
                ax.set_title(col_titles[col_idx], fontsize=6.3, pad=2, color=title_colors[col_idx], fontweight="bold")
            if col_idx == 0:
                ax.text(
                    0.015,
                    0.12,
                    FIELD_LABELS[field],
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.1,
                    color="white",
                    fontweight="bold",
                    bbox=dict(facecolor=FIELD_COLORS[field], edgecolor="none", boxstyle="round,pad=0.14", alpha=0.90),
                )


def write_contract(
    output_dir: Path,
    specs: list[DemoSpec],
    configs: dict[str, dict],
    final_metrics: dict[str, dict],
    extra_sources: list[Path],
) -> None:
    lines = [
        "# Figure Contract",
        "",
        "## Core Scientific Claim",
        "",
        (
            "Sparse-observation field selection and conditioning design materially affect "
            "turbulent-combustion field generation, and the N0 FNO baseline reconstructs "
            "worse than the GL-rbf point-cloud models for the evaluated snapshot."
        ),
        "",
        "## Archetype",
        "",
        "Asymmetric mixed-modality figure: configuration table, quantitative ablation plots, and a qualitative field plate.",
        "",
        "## Source Files Used",
        "",
    ]
    source_paths = []
    for spec in specs:
        source_paths.extend(
            [
                spec.config_json,
                spec.config_yaml,
                spec.loss_csv,
                metric_path(spec, spec.final_epoch, 2),
                metric_path(spec, spec.final_epoch, 4),
                metric_path(spec, spec.final_epoch, 8),
                *[field_image_path(spec, spec.final_epoch, 8, field) for field in FIELD_NAMES],
            ]
        )
    source_paths.extend(extra_sources)
    for path in sorted({p.as_posix() for p in source_paths}):
        lines.append(f"- `{path}`")

    lines.extend(
        [
            "",
            "## Panel Map",
            "",
            "- a. Configuration/evidence table for N0, N15, N17, and N18.",
            "- b. Saved checkpoint-time mean relative L2 trend at NFE=8.",
            "- c. Final field-wise relative L2 errors for CH4, CO, T, U_1, and p.",
            "- d. Final mean relative L2 sensitivity to NFE=2, 4, and 8.",
            "- e. Multi-field qualitative plate: CH4, CO, T, U_1, and p ground-truth strips plus reconstructions from each model.",
            "",
            "## Metrics And Statistics",
            "",
            "Metric: per-field normalized/relative L2 from existing `euler_nfe*_metrics.json` files.",
            "Aggregation: arithmetic mean across the five physical fields for panels b and d.",
            "Scope: saved visualization snapshot index 0; no new inference or training was run.",
            "",
            "## Final Snapshot Metrics",
            "",
            "| demo | backbone | conditioned fields | final epoch | mean relative L2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for spec in specs:
        cfg = configs[spec.key]
        fields = field_list_label(cfg.get("vis_cond_fields") or cfg.get("cond_fields") or [])
        lines.append(
            f"| {spec.short_label} | {cfg.get('backbone')} | {fields} | "
            f"{spec.final_epoch} | {mean_l2(final_metrics[spec.key]):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- N17 and N18 currently use checkpoint-time visualization metrics, not a full held-out dataset summary.",
            "- The qualitative panel embeds existing raster PNG evaluation outputs inside the SVG/PDF figure.",
            "- Ground-truth strips come from saved evaluation PNGs and may include green sparse-sensor overlays when that field was conditioned in the source evaluation image.",
            "- Existing full-dataset summaries were found for an older N15 run and an older latent-FM baseline, but they are not mixed into the main comparison because they are not the requested trained artifacts.",
        ]
    )
    (output_dir / "figure_contract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_caption(output_dir: Path) -> None:
    caption = """# Caption Draft

Sparse-observation conditioning controls turbulent-combustion field generation quality. a, Configuration summary for the FNO baseline (N0) and three GL-rbf point-cloud models (N15, N17, N18), showing the conditioned physical fields and sensor counts used for the saved evaluations. b, Checkpoint-time mean relative L2 error on the saved visualization snapshot, computed from NFE=8 evaluation artifacts. c, Final per-field relative L2 errors show that the FNO baseline has consistently larger reconstruction error, while GL-rbf performance depends on which fields are observed and conditioned. d, Final mean relative L2 remains ordered similarly across NFE=2, 4, and 8. e, Multi-field ground-truth and reconstruction matrix for CH4, CO, T, U1, and p, assembled from existing evaluation images, shows how model differences appear across chemically and hydrodynamically distinct variables. No training or new inference was run for this figure.
"""
    (output_dir / "caption_draft.md").write_text(caption, encoding="utf-8")


def build_figure(root: Path, output_dir: Path) -> None:
    configure_matplotlib()
    specs = build_specs(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        require_file(spec.config_json)
        require_file(spec.config_yaml)
        require_file(spec.loss_csv)
        require_file(metric_path(spec, spec.final_epoch, 8))
        for field in FIELD_NAMES:
            require_file(field_image_path(spec, spec.final_epoch, 8, field))
        require_file(spec.model_dir / "best.pt")
        require_file(spec.model_dir / "last.pt")
        require_file(spec.model_dir / "dataset_stats.pt")

    configs = {spec.key: load_json(spec.config_json) for spec in specs}
    final_metrics = {
        spec.key: numeric_field_metrics(load_json(metric_path(spec, spec.final_epoch, 8)))
        for spec in specs
    }

    fig = plt.figure(figsize=(7.2, 10.2), constrained_layout=False)
    outer = gridspec.GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[1.05, 1.65, 1.45, 4.75],
        width_ratios=[1.0, 1.0],
        hspace=0.50,
        wspace=0.30,
    )

    ax_table = fig.add_subplot(outer[0, :])
    plot_config_table(ax_table, specs, configs, final_metrics)

    ax_trend = fig.add_subplot(outer[1, 0])
    plot_training_trend(ax_trend, specs)

    ax_bars = fig.add_subplot(outer[1, 1])
    plot_final_bars(ax_bars, specs, final_metrics)

    ax_nfe = fig.add_subplot(outer[2, :])
    plot_nfe_sensitivity(ax_nfe, specs)

    plot_multifield_reconstruction_plate(fig, outer[3, :], specs)

    fig.align_ylabels()

    base = output_dir / FIGURE_NAME
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    extra_sources = [
        root / "0_demo_TurbulentCombustion/src/evaluate_ffm.py",
        root / "0_demo_TurbulentCombustion/src/evaluate_full_dataset.py",
        root / "0_demo_TurbulentCombustion/src/helpers.py",
    ]
    write_contract(output_dir, specs, configs, final_metrics, extra_sources)
    write_caption(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    build_figure(root, output_dir)
    print(f"Wrote figure package to {output_dir}")


if __name__ == "__main__":
    main()

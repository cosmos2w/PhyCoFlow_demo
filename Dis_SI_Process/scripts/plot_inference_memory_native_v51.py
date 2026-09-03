#!/usr/bin/env python
"""Plot the standardized V5.1 model-to-native-inference memory comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MM = 1.0 / 25.4
METHOD_ORDER = [
    "DMF-Gen",
    "Senseiver",
    "SiT",
    "Geo-FNO",
    "FFM-Perceiver",
    "FFM-FNO",
    "MLP-RBF",
    "Latent FM",
]
COLORS = {
    "DMF-Gen": "#E63946",
    "FFM-FNO": "#1D3557",
    "FFM-Perceiver": "#457B9D",
    "Latent FM": "#6A4C93",
    "SiT": "#A28BC4",
    "MLP-RBF": "#2A9D8F",
    "Geo-FNO": "#E9A03B",
    "Senseiver": "#2F6F8F",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "font.size": 6.2,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.0,
            "legend.frameon": False,
        }
    )


def make_figure(data: pd.DataFrame) -> plt.Figure:
    data = data.set_index("method").loc[METHOD_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(183 * MM, 78 * MM))
    y = np.arange(len(data))

    for index, row in data.iterrows():
        method = str(row.method)
        color = COLORS[method]
        model = float(row.model_state_mib)
        peak = float(row.inference_peak_allocated_mib)
        ax.hlines(index, model, peak, color=color, linewidth=2.0, alpha=0.42, zorder=2)
        ax.plot(model, index, marker="o", markersize=4.3, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.0, zorder=3)
        ax.plot(peak, index, marker="o", markersize=5.0, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.65, zorder=4)
        ax.text(model, index - 0.28, f"{model:.0f}", color=color, fontsize=5.1, ha="center", va="top")
        ax.text(peak, index + 0.28, f"{peak:.0f}", color=color, fontsize=5.1, ha="center", va="bottom")

    ax.set_xscale("log")
    low = max(0.5, float(data.model_state_mib.min()) / 1.6)
    high = float(data.inference_peak_allocated_mib.max()) * 1.55
    ax.set_xlim(low, high)
    ax.set_ylim(len(data) - 0.42, -0.62)
    ax.set_yticks(y, data.method)
    for tick, method in zip(ax.get_yticklabels(), data.method):
        tick.set_color(COLORS[str(method)])
        tick.set_fontweight("semibold" if method == "DMF-Gen" else "normal")
    ax.set_xlabel("GPU memory (MiB, log scale)", fontsize=6.5)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)

    ax.plot([], [], marker="o", markersize=4.3, markerfacecolor="white", markeredgecolor="#444444", linestyle="none", label="Model parameters + buffers")
    ax.plot([], [], marker="o", markersize=5.0, markerfacecolor="#666666", markeredgecolor="white", linestyle="none", label="Peak allocated during inference")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.015), ncol=2, fontsize=5.6, handletextpad=0.5, columnspacing=1.2)

    fig.suptitle("From model state to native inference memory", x=0.125, y=0.985, ha="left", fontsize=9.2, fontweight="semibold")
    fig.text(
        0.125,
        0.925,
        r"$B=1$ · $M=256$ · $N=40{,}300$ · float32 · uniform torch.inference_mode()",
        ha="left",
        va="top",
        fontsize=6.0,
        color="#555555",
    )
    fig.text(
        0.125,
        0.018,
        "Peak includes fixed device inputs and persistent inference cache; DMF uses its adopted streamed path, while Senseiver is currently unstreamed.",
        ha="left",
        va="bottom",
        fontsize=5.2,
        color="#555555",
    )
    fig.text(0.018, 0.955, "D6", ha="left", va="top", fontsize=8.2, fontweight="bold")
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.19, top=0.80)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--preview", type=Path, help="Optional temporary Python-rendered PNG for visual QA")
    args = parser.parse_args()

    data = pd.read_csv(args.source)
    data = data.loc[data.status == "ok"].copy()
    missing = sorted(set(METHOD_ORDER) - set(data.method))
    if missing:
        raise RuntimeError(f"Missing successful methods: {missing}")
    if not (data.inference_context == "torch.inference_mode").all():
        raise RuntimeError("Refusing to plot a mixed inference-context source")
    if data.output_requires_grad.astype(bool).any():
        raise RuntimeError("Refusing to plot outputs carrying autograd state")

    configure_style()
    fig = make_figure(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg", bbox_inches=None)
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, format="png", dpi=220, bbox_inches=None)
    plt.close(fig)

    text = args.output.read_text(encoding="utf-8")
    qa = {
        "status": "pass",
        "svg_exists": args.output.is_file(),
        "svg_bytes": args.output.stat().st_size,
        "editable_text": "<text" in text,
        "viewbox_present": "viewBox" in text,
        "method_count": int(data.method.nunique()),
        "all_methods_present": set(data.method) == set(METHOD_ORDER),
        "uniform_inference_context": bool((data.inference_context == "torch.inference_mode").all()),
        "no_autograd_outputs": not bool(data.output_requires_grad.astype(bool).any()),
        "output": str(args.output),
        "source": str(args.source),
    }
    args.qa.parent.mkdir(parents=True, exist_ok=True)
    args.qa.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

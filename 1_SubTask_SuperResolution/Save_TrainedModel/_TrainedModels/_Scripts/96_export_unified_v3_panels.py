#!/usr/bin/env python
"""Export unified-v3 standalone panels, SI figures, and LaTeX tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

import global_style as manuscript
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import apply_style, finalize_colorbar_multiplier_alignment, save_figure
from common.io_utils import matching_or_latest, read_csv, write_json
from common.multiscale_wavelet_panels import draw_multiscale_components, draw_multiscale_fidelity, draw_sensor_efficiency
from common.physical_figure_layout import create_standalone_canvas, resolve_physical_layout
from common.publication_panels import PublicationContext, _float, _int
from common.publication_panels_unified_v2 import draw_panel_c as draw_v2_panel_c
from common.publication_panels_unified_v3 import PANEL_OUTPUT_NAMES, draw_panel, panel_label


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _deep_merge(merged[key], value) if key in merged else value
    return merged


def load_layout(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    return _deep_merge(load_layout((path.resolve().parent / parent).resolve()), payload)


def sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path):
    path = Path(path)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def make_context(args, cfg, layout, rid):
    cache_manifest = args.cache_manifest or (
        RESULTS_DIR / "ReconstructionCache" / "ReconstructionCache_manifest_formal_20260712.csv"
    )
    representatives = args.representatives or matching_or_latest(
        RESULTS_DIR / "CanonicalTestIndex", "RepresentativeSnapshots", args.base_data_run_id, "csv",
    )
    source_run_ids = {
        "ResolutionProtocol_fields": args.base_data_run_id,
        "ResolutionProtocol_budgets": args.base_data_run_id,
        "ResolutionProtocol_sensors": args.base_data_run_id,
        "SensorSweepAllRecipes_summary": args.data_run_id,
        "MultiscaleWavelet_summary": args.multiscale_run_id,
        "MultiscaleWavelet_metadata": args.multiscale_run_id,
    }
    ctx = PublicationContext(cfg, args.data_run_id, Path(cache_manifest), Path(representatives), source_run_ids)
    ctx.v2 = layout
    return ctx


def finish_figure(fig):
    typography = manuscript.enforce_figure_typography(fig)
    finalize_colorbar_multiplier_alignment(fig)
    target = .75
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(target)
        ax.tick_params(axis="both", which="both", width=target)
    return typography


def save_triplet(fig, base: Path, cfg):
    outputs = save_figure(fig, base, cfg, formats=("svg", "pdf", "png"),
                          dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None)
    plt.close(fig)
    return [Path(path) for path in outputs]


def export_standalone_panels(ctx, cfg, layout, physical, root):
    outputs, metadata = [], {}
    panel_root = root / "panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    for label in "abcd":
        fig, ax, _container = create_standalone_canvas(physical, label)
        panel_label(ax, label)
        metadata[label] = draw_panel(label, ax, ctx)
        finish_figure(fig)
        base = panel_root / f"{PANEL_OUTPUT_NAMES[label]}_{root.name.rsplit('_', 2)[-2]}_{root.name.rsplit('_', 1)[-1]}"
        outputs.extend(save_triplet(fig, base, cfg))
    return outputs, metadata


def _blank_parent(width_mm, height_mm):
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), layout=None)
    parent = fig.add_axes([.025, .01, .965, .98], frameon=False)
    parent.set_axis_off()
    return fig, parent


def export_si(ctx, cfg, layout, root, rid):
    si_root = root / "si"
    si_root.mkdir(parents=True, exist_ok=True)
    records, metadata = {}, {}

    width, height = map(float, layout["si_streamlined"]["sensor_sweep_size_mm"])
    fig, parent = _blank_parent(width, height)
    metadata["Sx1"] = draw_sensor_efficiency(parent, ctx, standalone=True, show_legend=True)
    finish_figure(fig)
    records["Sx1_complete_sensor_sweeps"] = save_triplet(
        fig, si_root / f"SI_Sx1_complete_sensor_sweeps_{rid}", cfg,
    )

    width, height = map(float, layout["si_streamlined"]["qualitative_gallery_size_mm"])
    fig, parent = _blank_parent(width, height)
    metadata["Sx2"] = draw_v2_panel_c(parent, ctx, standalone=False, show_legend=False, version=2)
    finish_figure(fig)
    records["Sx2_complete_recipe_gallery"] = save_triplet(
        fig, si_root / f"SI_Sx2_complete_recipe_gallery_{rid}", cfg,
    )

    width, height = map(float, layout["si_streamlined"]["wavelet_size_mm"])
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), layout=None)
    left = fig.add_axes([.018, .01, .442, .98], frameon=False); left.set_axis_off()
    right = fig.add_axes([.475, .01, .515, .98], frameon=False); right.set_axis_off()
    metadata["Sx3_components"] = draw_multiscale_components(left, ctx, standalone=False, show_legend=False)
    metadata["Sx3_heatmaps"] = draw_multiscale_fidelity(right, ctx, standalone=False, show_legend=False)
    finish_figure(fig)
    records["Sx3_complete_three_scale_wavelet"] = save_triplet(
        fig, si_root / f"SI_Sx3_complete_three_scale_wavelet_{rid}", cfg,
    )
    return records, metadata


def latex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_latex_table(path: Path, columns, rows, caption: str, label: str):
    aligns = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{aligns}}}", r"\toprule",
        " & ".join(latex_escape(column) for column in columns) + r" \\", r"\midrule",
    ]
    lines.extend(" & ".join(latex_escape(value) for value in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_tables(layout, root, rid):
    table_root = root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    sensor_path = RESULTS_DIR / "UnifiedPublicationV2" / "SensorSweepAllRecipes_summary_20260806_1124.csv"
    wavelet_path = RESULTS_DIR / "MultiscaleWavelet" / "MultiscaleWavelet_summary_20260802_1250.csv"
    sensor_rows = [row for row in read_csv(sensor_path) if row["metric"] == "physical_rel_l2"]
    wavelet_rows = read_csv(wavelet_path)
    model_order = ["DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF"]
    recipe_order = list(layout["panel_b_streamlined"]["recipes"])
    scale_order = ["large", "intermediate", "fine"]
    count_order = [64, 128, 256, 384, 512]
    outputs = {}

    accuracy = []
    for model in model_order:
        for recipe in recipe_order:
            row = next(row for row in sensor_rows if row["model"] == model and row["recipe"] == recipe
                       and _int(row["sensor_count"]) == 512)
            accuracy.append([row["model_label"], row["recipe_label"], "512", f"{_float(row['mean']):.6f}",
                             f"{_float(row['ci95_low']):.6f}", f"{_float(row['ci95_high']):.6f}", row["valid_n"]])
    path = table_root / f"accuracy_512_{rid}.tex"
    write_latex_table(path, ["Method", "Recipe", "Sensors", "Mean", "CI low", "CI high", "n"], accuracy,
                      "Physical relative-$L_2$ at 512 sensors.", "tab:mixed-resolution-accuracy-512")
    outputs["accuracy_512"] = {"path": path, "row_count": len(accuracy)}

    sweeps = []
    for model in model_order:
        for recipe in recipe_order:
            for count in count_order:
                row = next(row for row in sensor_rows if row["model"] == model and row["recipe"] == recipe
                           and _int(row["sensor_count"]) == count)
                sweeps.append([row["model_label"], row["recipe_label"], count, f"{_float(row['mean']):.6f}",
                               f"{_float(row['ci95_low']):.6f}", f"{_float(row['ci95_high']):.6f}", row["valid_n"]])
    path = table_root / f"sensor_sweeps_64_512_{rid}.tex"
    write_latex_table(path, ["Method", "Recipe", "Sensors", "Mean", "CI low", "CI high", "n"], sweeps,
                      "Physical relative-$L_2$ sensor sweeps.", "tab:mixed-resolution-sensor-sweeps")
    outputs["sensor_sweeps_64_512"] = {"path": path, "row_count": len(sweeps)}

    for metric, stem, caption, label in (
        ("pattern_correlation", "pattern_correlations_all_scales", "Multiscale spatial pattern correlations.",
         "tab:mixed-resolution-wavelet-correlation"),
        ("variance_fraction_bias_pp", "variance_allocation_bias_all_scales",
         "Multiscale variance-allocation biases in percentage points.",
         "tab:mixed-resolution-wavelet-bias"),
    ):
        body = []
        for model in model_order:
            for recipe in recipe_order:
                for scale in scale_order:
                    row = next(row for row in wavelet_rows if row["model_key"] == model
                               and row["recipe"] == recipe and row["scale_group"] == scale
                               and row["metric"] == metric)
                    body.append([row["model_label"], row["recipe_label"], scale,
                                 f"{_float(row['median']):.9f}", row["valid_n"]])
        path = table_root / f"{stem}_{rid}.tex"
        write_latex_table(path, ["Method", "Recipe", "Scale", "Median", "n"], body, caption, label)
        outputs[stem] = {"path": path, "row_count": len(body)}
    return outputs, [sensor_path, wavelet_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=Path(__file__).with_name("publication_layout_unified_v3.yaml"))
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    rid = args.run_id
    if not rid:
        raise ValueError("--run-id is required for additive v3 exports")
    cfg = load_config(args.config); manuscript.register_local_arial(); apply_style(cfg); ensure_output_dirs()
    layout = load_layout(args.layout); physical = resolve_physical_layout(layout)
    ctx = make_context(args, cfg, layout, rid)
    root = FIGURES_DIR / "UnifiedV3" / rid
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing v3 export directory: {root}")
    root.mkdir(parents=True)
    panel_outputs, panel_meta = export_standalone_panels(ctx, cfg, layout, physical, root)
    si_outputs, si_meta = export_si(ctx, cfg, layout, root, rid)
    table_outputs, table_sources = export_tables(layout, root, rid)
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v3_{rid}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["standalone_outputs"] = [record(path) for path in panel_outputs]
    manifest["standalone_panel_metadata"] = panel_meta
    manifest["si_outputs"] = {
        key: [record(path) for path in paths] for key, paths in si_outputs.items()
    }
    manifest["si_metadata"] = si_meta
    manifest["table_outputs"] = {
        key: {**{k: v for k, v in item.items() if k != "path"}, **record(item["path"])}
        for key, item in table_outputs.items()
    }
    manifest["table_sources"] = [record(path) for path in table_sources]
    manifest["si_root"] = str(root.resolve())
    write_json(manifest_path, manifest)
    print(f"[OK] {root}")


if __name__ == "__main__":
    main()

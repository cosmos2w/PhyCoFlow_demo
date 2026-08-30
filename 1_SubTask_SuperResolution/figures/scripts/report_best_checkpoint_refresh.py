#!/usr/bin/env python
"""Compare the refreshed best-checkpoint results with the unified-v2 baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DISPLAY_RECIPES = ["3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"]


def _merge(old_path: Path, new_path: Path, keys: list[str]) -> pd.DataFrame:
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    value_columns = ["mean", "median", "q25", "q75", "ci95_low", "ci95_high"]
    keep = keys + [column for column in value_columns if column in old and column in new]
    merged = old[keep].merge(new[keep], on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    for column in value_columns:
        if f"{column}_old" not in merged:
            continue
        merged[f"{column}_delta"] = merged[f"{column}_new"] - merged[f"{column}_old"]
        merged[f"{column}_pct_change"] = 100.0 * merged[f"{column}_delta"] / merged[f"{column}_old"]
    return merged


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-sweep", type=Path, required=True)
    parser.add_argument("--new-sweep", type=Path, required=True)
    parser.add_argument("--old-wavelet", type=Path, required=True)
    parser.add_argument("--new-wavelet", type=Path, required=True)
    parser.add_argument("--old-sweep-snapshots", type=Path, required=True)
    parser.add_argument("--new-sweep-snapshots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-sensor-count", type=int, default=512)
    parser.add_argument("--qualitative-snapshot", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep_keys = ["model", "model_label", "recipe", "recipe_label", "sensor_count", "metric"]
    sweep = _merge(args.old_sweep, args.new_sweep, sweep_keys)
    sweep.to_csv(args.output_dir / "sensor_sweep_changes.csv", index=False)

    fixed = sweep[sweep["sensor_count"].eq(args.fixed_sensor_count)].copy()
    fixed.to_csv(args.output_dir / f"fixed_{args.fixed_sensor_count}_sensor_changes.csv", index=False)

    fixed_model = fixed.groupby(["model", "model_label"], as_index=False).agg(
        mean_old=("mean_old", "mean"),
        mean_new=("mean_new", "mean"),
        improved_cells=("mean_delta", lambda values: int((values < 0).sum())),
        cells=("mean_delta", "size"),
    )
    fixed_model["mean_delta"] = fixed_model["mean_new"] - fixed_model["mean_old"]
    fixed_model["mean_pct_change"] = 100.0 * fixed_model["mean_delta"] / fixed_model["mean_old"]
    fixed_model.to_csv(args.output_dir / f"fixed_{args.fixed_sensor_count}_sensor_model_summary.csv", index=False)

    grid_model = sweep.groupby(["model", "model_label"], as_index=False).agg(
        median_cell_pct_change=("mean_pct_change", "median"),
        mean_cell_pct_change=("mean_pct_change", "mean"),
        improved_cells=("mean_delta", lambda values: int((values < 0).sum())),
        cells=("mean_delta", "size"),
    )
    grid_model.to_csv(args.output_dir / "all_sensor_recipe_model_summary.csv", index=False)

    wavelet_keys = [
        "model_key", "model", "model_label", "recipe", "recipe_label",
        "scale_group", "scale_label", "metric",
    ]
    wavelet = _merge(args.old_wavelet, args.new_wavelet, wavelet_keys)
    wavelet.to_csv(args.output_dir / "wavelet_changes.csv", index=False)
    displayed = wavelet[wavelet["recipe"].isin(DISPLAY_RECIPES)].copy()

    fine_corr = displayed[
        displayed["metric"].eq("pattern_correlation") & displayed["scale_group"].eq("fine")
    ]
    fine_model = fine_corr.groupby(["model_key", "model_label"], as_index=False).agg(
        median_old=("median_old", "mean"), median_new=("median_new", "mean")
    )
    fine_model["median_delta"] = fine_model["median_new"] - fine_model["median_old"]
    fine_model.to_csv(args.output_dir / "fine_scale_correlation_model_summary.csv", index=False)

    bias = displayed[displayed["metric"].eq("variance_fraction_bias_pp")].copy()
    bias["abs_median_old"] = bias["median_old"].abs()
    bias["abs_median_new"] = bias["median_new"].abs()
    bias_model = bias.groupby(["model_key", "model_label"], as_index=False).agg(
        mean_abs_bias_old=("abs_median_old", "mean"),
        mean_abs_bias_new=("abs_median_new", "mean"),
    )
    bias_model["mean_abs_bias_delta"] = bias_model["mean_abs_bias_new"] - bias_model["mean_abs_bias_old"]
    bias_model.to_csv(args.output_dir / "variance_bias_model_summary.csv", index=False)

    old_snapshot = pd.read_csv(args.old_sweep_snapshots)
    new_snapshot = pd.read_csv(args.new_sweep_snapshots)
    snapshot_keys = ["model", "model_label", "recipe", "recipe_label", "snapshot_index", "sensor_count"]
    snapshot = old_snapshot.merge(
        new_snapshot,
        on=snapshot_keys,
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    snapshot = snapshot[
        snapshot["snapshot_index"].eq(args.qualitative_snapshot)
        & snapshot["sensor_count"].eq(args.fixed_sensor_count)
        & snapshot["recipe"].isin(DISPLAY_RECIPES)
        & snapshot["model"].isin(["DMFGen", "FFM_Perceiver", "Senseiver"])
    ].copy()
    snapshot["physical_rel_l2_delta"] = snapshot["physical_rel_l2_new"] - snapshot["physical_rel_l2_old"]
    snapshot["physical_rel_l2_pct_change"] = (
        100.0 * snapshot["physical_rel_l2_delta"] / snapshot["physical_rel_l2_old"]
    )
    snapshot.to_csv(args.output_dir / "qualitative_snapshot_changes.csv", index=False)

    best = fixed.nsmallest(5, "mean_pct_change")
    worst = fixed[fixed["mean_pct_change"] > 0].nlargest(5, "mean_pct_change")
    lines = [
        "# Best-checkpoint refresh: quantitative changes",
        "",
        f"Baseline: `{args.old_sweep.name}` / `{args.old_wavelet.name}`.",
        f"Refresh: `{args.new_sweep.name}` / `{args.new_wavelet.name}`.",
        "Negative relative-L2 change is an improvement; positive correlation change is an improvement.",
        "",
        f"## Fixed {args.fixed_sensor_count}-sensor mean physical relative L2",
        "",
        "| Model | Baseline | Refreshed | Change | Improved recipes |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fixed_model.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {_fmt(row.mean_old)} | {_fmt(row.mean_new)} | "
            f"{row.mean_pct_change:+.1f}% | {row.improved_cells}/{row.cells} |"
        )
    lines.extend([
        "",
        "## Across the full 5-recipe × 5-sensor grid",
        "",
        "| Model | Median cell change | Mean cell change | Improved cells |",
        "|---|---:|---:|---:|",
    ])
    for row in grid_model.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {row.median_cell_pct_change:+.1f}% | "
            f"{row.mean_cell_pct_change:+.1f}% | {row.improved_cells}/{row.cells} |"
        )
    lines.extend(["", f"## Largest {args.fixed_sensor_count}-sensor improvements", ""])
    for row in best.itertuples(index=False):
        lines.append(f"- {row.model_label}, {row.recipe_label}: {row.mean_pct_change:+.1f}% ({row.mean_old:.4f} → {row.mean_new:.4f})")
    lines.extend(["", f"## Largest {args.fixed_sensor_count}-sensor regressions", ""])
    for row in worst.itertuples(index=False):
        lines.append(f"- {row.model_label}, {row.recipe_label}: {row.mean_pct_change:+.1f}% ({row.mean_old:.4f} → {row.mean_new:.4f})")
    lines.extend([
        "",
        "## Displayed-recipe fine-scale spatial correlation",
        "",
        "| Model | Baseline | Refreshed | Absolute change |",
        "|---|---:|---:|---:|",
    ])
    for row in fine_model.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {row.median_old:.3f} | {row.median_new:.3f} | {row.median_delta:+.3f} |"
        )
    lines.extend([
        "",
        "## Displayed-recipe mean absolute variance-allocation bias",
        "",
        "| Model | Baseline (pp) | Refreshed (pp) | Change (pp) |",
        "|---|---:|---:|---:|",
    ])
    for row in bias_model.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {row.mean_abs_bias_old:.3f} | {row.mean_abs_bias_new:.3f} | "
            f"{row.mean_abs_bias_delta:+.3f} |"
        )
    lines.extend([
        "",
        f"## Qualitative snapshot {args.qualitative_snapshot} at {args.fixed_sensor_count} sensors",
        "",
        "| Model | Recipe | Baseline L2 | Refreshed L2 | Change |",
        "|---|---|---:|---:|---:|",
    ])
    for row in snapshot.sort_values(["model", "recipe"]).itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {row.recipe_label} | {row.physical_rel_l2_old:.4f} | "
            f"{row.physical_rel_l2_new:.4f} | {row.physical_rel_l2_pct_change:+.1f}% |"
        )
    (args.output_dir / "quantitative_change_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

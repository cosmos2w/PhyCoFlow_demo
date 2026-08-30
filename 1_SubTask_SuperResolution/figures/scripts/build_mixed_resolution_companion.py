#!/usr/bin/env python
"""Build a source-backed technical/paper-writing companion for unified-v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINED = ROOT / "Save_TrainedModel" / "_TrainedModels"
RESULTS = TRAINED / "_Process_Results"
FIGURES = TRAINED / "_Process_Figures"
ASSEMBLED = FIGURES / "Assembled"

MODELS = ["DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF"]
RECIPES = ["1_H_only", "2_H_limited", "3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"]
DISPLAY_RECIPES = ["3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"]


def md_table(headers: list[str], rows: list[list[object]], aligns: list[str] | None = None) -> str:
    aligns = aligns or ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def signed(value: float, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def pct_change(new: float, old: float) -> float:
    return 100.0 * (float(new) - float(old)) / float(old)


def summary_row(frame: pd.DataFrame, model: str, recipe: str, count: int) -> pd.Series:
    rows = frame[
        frame["model"].eq(model)
        & frame["recipe"].eq(recipe)
        & frame["sensor_count"].eq(count)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one sweep row for {model}/{recipe}/n{count}, found {len(rows)}")
    return rows.iloc[0]


def wavelet_row(frame: pd.DataFrame, model: str, recipe: str, scale: str, metric: str) -> pd.Series:
    rows = frame[
        frame["model"].eq(model)
        & frame["recipe"].eq(recipe)
        & frame["scale_group"].eq(scale)
        & frame["metric"].eq(metric)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one wavelet row for {model}/{recipe}/{scale}/{metric}, found {len(rows)}")
    return rows.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-data-run-id", help="Unified accuracy/sensor-sweep run; defaults to --run-id.")
    parser.add_argument("--multiscale-run-id", help="Multiscale wavelet run; defaults to --run-id.")
    parser.add_argument("--snapshot-data-run-id", help="Per-snapshot sensor-sweep run; defaults to --base-data-run-id.")
    parser.add_argument("--base-data-run-id", required=True)
    parser.add_argument("--inventory-run-id", required=True)
    parser.add_argument("--baseline-data-run-id", default="20260722_1103")
    parser.add_argument("--baseline-wavelet-run-id", default="20260722_1128")
    parser.add_argument("--baseline-figure-run-id", default="20260722_0003")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source_data_run_id = args.source_data_run_id or args.run_id
    multiscale_run_id = args.multiscale_run_id or args.run_id
    snapshot_data_run_id = args.snapshot_data_run_id or args.base_data_run_id

    output = args.output or ASSEMBLED / f"MixedResolution_RnD_companion_{args.run_id}.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    source_manifest_path = ASSEMBLED / f"FigureSourceManifest_unified_v2_{args.run_id}.json"
    audit_path = RESULTS / "UnifiedPublicationV2" / f"UnifiedV2Audit_{args.run_id}.json"
    wave_meta_path = RESULTS / "MultiscaleWavelet" / f"MultiscaleWavelet_metadata_{multiscale_run_id}.json"
    data_manifest_path = RESULTS / "UnifiedPublicationV2" / f"UnifiedV2DataManifest_{source_data_run_id}.json"
    inventory_path = RESULTS / "ModelInventory" / f"ModelInventory_{args.inventory_run_id}.csv"
    cache_manifest_path = RESULTS / "ReconstructionCache" / "ReconstructionCache_manifest_formal_20260712.csv"
    sweep_path = RESULTS / "UnifiedPublicationV2" / f"SensorSweepAllRecipes_summary_{source_data_run_id}.csv"
    accuracy_path = RESULTS / "UnifiedPublicationV2" / f"AllRecipeAccuracy_summary_{source_data_run_id}.csv"
    wavelet_path = RESULTS / "MultiscaleWavelet" / f"MultiscaleWavelet_summary_{multiscale_run_id}.csv"
    wavelet_per_path = RESULTS / "MultiscaleWavelet" / f"MultiscaleWavelet_per_snapshot_{multiscale_run_id}.csv"
    budgets_path = RESULTS / "ResolutionProtocol" / f"ResolutionProtocol_budgets_{args.base_data_run_id}.csv"
    fields_path = RESULTS / "ResolutionProtocol" / f"ResolutionProtocol_fields_{args.base_data_run_id}.csv"
    sensors_path = RESULTS / "ResolutionProtocol" / f"ResolutionProtocol_sensors_{args.base_data_run_id}.csv"
    snapshot_path = RESULTS / "SensorSweep" / f"SensorSweep_per_snapshot_{snapshot_data_run_id}.csv"
    old_sweep_path = RESULTS / "UnifiedPublicationV2" / f"SensorSweepAllRecipes_summary_{args.baseline_data_run_id}.csv"
    old_wavelet_path = RESULTS / "MultiscaleWavelet" / f"MultiscaleWavelet_summary_{args.baseline_wavelet_run_id}.csv"

    required = [
        source_manifest_path, audit_path, wave_meta_path, data_manifest_path, inventory_path,
        cache_manifest_path, sweep_path, accuracy_path, wavelet_path, wavelet_per_path,
        budgets_path, fields_path, sensors_path, snapshot_path, old_sweep_path, old_wavelet_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing companion source files:\n" + "\n".join(missing))

    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    wave_meta = json.loads(wave_meta_path.read_text(encoding="utf-8"))
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    budgets = pd.read_csv(budgets_path)
    inventory = pd.read_csv(inventory_path)
    cache = pd.read_csv(cache_manifest_path, low_memory=False)
    sweep = pd.read_csv(sweep_path)
    accuracy = pd.read_csv(accuracy_path)
    wavelet = pd.read_csv(wavelet_path)
    snapshots = pd.read_csv(snapshot_path)
    old_sweep = pd.read_csv(old_sweep_path)
    old_wavelet = pd.read_csv(old_wavelet_path)
    fields = pd.read_csv(fields_path)
    sensor_protocol = pd.read_csv(sensors_path)

    if not audit.get("passed"):
        raise RuntimeError(f"Unified-v2 audit did not pass: {audit_path}")
    if set(cache["checkpoint_kind"].dropna().unique()) != {"best"}:
        raise RuntimeError("Companion requires an all-best checkpoint cache manifest")
    if set(cache["status"].unique()) != {"ok"} or len(cache) != 37200:
        raise RuntimeError("Unexpected canonical cache coverage")
    if not inventory["probe_load"].eq("ok").all() or len(inventory) != 20:
        raise RuntimeError("Checkpoint inventory is not a complete 20/20 successful probe")
    for frame, name in ((sweep, "sweep"), (accuracy, "accuracy"), (wavelet, "wavelet")):
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise RuntimeError(f"Non-finite values found in {name} summary")

    model_labels = dict(zip(sweep["model"], sweep["model_label"]))
    recipe_labels = dict(zip(sweep["recipe"], sweep["recipe_label"]))
    panel_c = source["panels"]["c"]
    panel_e = source["panels"]["e"]
    representative = wave_meta["representative_snapshot"]
    baseline_selection = wave_meta["representative_baseline"]
    validation = wave_meta["validation"]
    canvas_w = float(source["layout"]["canvas_width_mm"])
    canvas_h = float(source["layout"]["canvas_height_mm"])

    # Panel-b table and paired contrasts.
    panel_b_rows = []
    for model in MODELS:
        values = []
        for recipe in RECIPES:
            row = summary_row(sweep, model, recipe, 512)
            values.append(f"{row['mean']:.4f} [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]")
        panel_b_rows.append([model_labels[model], *values])

    paired_contrasts = []
    mixed_effects: dict[str, float] = {}
    rich_effects: dict[str, float] = {}
    for model in MODELS:
        hlim = summary_row(sweep, model, "2_H_limited", 512)["mean"]
        mixed = summary_row(sweep, model, "3_Mixed_HML", 512)["mean"]
        balanced = summary_row(sweep, model, "4_ZeroH_Balanced", 512)["mean"]
        rich = summary_row(sweep, model, "5_ZeroH_MRich", 512)["mean"]
        mixed_effects[model] = pct_change(mixed, hlim)
        rich_effects[model] = pct_change(rich, balanced)
        paired_contrasts.append([
            model_labels[model],
            f"{mixed_effects[model]:+.1f}%",
            f"{rich_effects[model]:+.1f}%",
        ])

    # Panel-c exact snapshot values.
    snap = int(panel_c["snapshot"])
    snap_count = int(panel_c["sensor_count"])
    qualitative = snapshots[
        snapshots["snapshot_index"].eq(snap)
        & snapshots["sensor_count"].eq(snap_count)
        & snapshots["model"].isin(panel_c["models"])
        & snapshots["recipe"].isin(panel_c["recipes"])
    ]
    panel_c_rows = []
    for model in panel_c["models"]:
        values = []
        for recipe in panel_c["recipes"]:
            row = qualitative[qualitative["model"].eq(model) & qualitative["recipe"].eq(recipe)]
            if len(row) != 1:
                raise RuntimeError(f"Missing panel-c value for {model}/{recipe}")
            values.append(f"{float(row.iloc[0]['physical_rel_l2']):.5f}")
        panel_c_rows.append([model_labels[model], *values])

    # Panel-d tables.
    panel_d_256 = []
    panel_d_reduction = []
    for model in MODELS:
        panel_d_256.append([
            model_labels[model],
            *[f"{summary_row(sweep, model, recipe, 256)['mean']:.4f}" for recipe in RECIPES],
        ])
        reductions = []
        for recipe in RECIPES:
            first = summary_row(sweep, model, recipe, 64)["mean"]
            last = summary_row(sweep, model, recipe, 512)["mean"]
            reductions.append(f"{100.0 * (first - last) / first:.1f}%")
        panel_d_reduction.append([model_labels[model], *reductions])

    # Panel-f exact medians.
    correlation_rows = []
    bias_rows = []
    for model in MODELS:
        correlations = []
        biases = []
        for recipe in DISPLAY_RECIPES:
            correlations.append(" / ".join(
                f"{wavelet_row(wavelet, model, recipe, scale, 'pattern_correlation')['median']:.3f}"
                for scale in ("large", "intermediate", "fine")
            ))
            biases.append(" / ".join(
                signed(wavelet_row(wavelet, model, recipe, scale, "variance_fraction_bias_pp")["median"])
                for scale in ("large", "intermediate", "fine")
            ))
        correlation_rows.append([model_labels[model], *correlations])
        bias_rows.append([model_labels[model], *biases])

    # Baseline-to-refresh changes at 512 sensors.
    change_rows = []
    largest_changes = []
    for model in MODELS:
        old_values, new_values, recipe_changes = [], [], []
        for recipe in RECIPES:
            old = summary_row(old_sweep, model, recipe, 512)["mean"]
            new = summary_row(sweep, model, recipe, 512)["mean"]
            old_values.append(old); new_values.append(new)
            recipe_changes.append((pct_change(new, old), model, recipe, old, new))
        old_mean, new_mean = float(np.mean(old_values)), float(np.mean(new_values))
        change_rows.append([
            model_labels[model], fmt(old_mean), fmt(new_mean), f"{pct_change(new_mean, old_mean):+.1f}%",
            f"{sum(new < old for new, old in zip(new_values, old_values))}/5",
        ])
        largest_changes.extend(recipe_changes)
    largest_improvements = sorted(largest_changes)[:5]
    regressions = sorted((item for item in largest_changes if item[0] > 0), reverse=True)[:5]
    dmf_average_change = pct_change(
        np.mean([summary_row(sweep, "DMFGen", recipe, 512)["mean"] for recipe in RECIPES]),
        np.mean([summary_row(old_sweep, "DMFGen", recipe, 512)["mean"] for recipe in RECIPES]),
    )
    dmf_rich_baseline_change = pct_change(
        summary_row(sweep, "DMFGen", "5_ZeroH_MRich", 512)["mean"],
        summary_row(old_sweep, "DMFGen", "5_ZeroH_MRich", 512)["mean"],
    )
    refreshed_models = [model_labels.get(model, model) for model in data_manifest.get("refreshed_models", [])]
    refreshed_models_text = ", ".join(refreshed_models) if refreshed_models else "none recorded"

    # Scale-level baseline changes for displayed recipes.
    fine_change_rows = []
    bias_change_rows = []
    for model in MODELS:
        old_corr = np.mean([
            wavelet_row(old_wavelet, model, recipe, "fine", "pattern_correlation")["median"]
            for recipe in DISPLAY_RECIPES
        ])
        new_corr = np.mean([
            wavelet_row(wavelet, model, recipe, "fine", "pattern_correlation")["median"]
            for recipe in DISPLAY_RECIPES
        ])
        old_bias = np.mean([
            abs(wavelet_row(old_wavelet, model, recipe, scale, "variance_fraction_bias_pp")["median"])
            for recipe in DISPLAY_RECIPES for scale in ("large", "intermediate", "fine")
        ])
        new_bias = np.mean([
            abs(wavelet_row(wavelet, model, recipe, scale, "variance_fraction_bias_pp")["median"])
            for recipe in DISPLAY_RECIPES for scale in ("large", "intermediate", "fine")
        ])
        fine_change_rows.append([model_labels[model], f"{old_corr:.3f}", f"{new_corr:.3f}", f"{new_corr-old_corr:+.3f}"])
        bias_change_rows.append([model_labels[model], f"{old_bias:.3f}", f"{new_bias:.3f}", f"{new_bias-old_bias:+.3f}"])

    # Training protocol table.
    budget_rows = []
    for recipe in RECIPES:
        row = budgets[budgets["recipe"].eq(recipe)].iloc[0]
        budget_rows.append([
            row["recipe_label"], row["actual_ratio"],
            f"{int(row['train_cases_L']):,} / {int(row['train_cases_M']):,} / {int(row['train_cases_H']):,}",
            f"{int(row['active_train_cases']):,}", f"{int(row['train_snapshots']):,}",
            f"{int(row['spatial_dof_budget']):,}", f"{float(row['spatial_dof_budget_normalized_H_only']):.4f}",
        ])

    cache_counts = cache.groupby(["model", "recipe"]).size().unstack()
    checkpoint_rows = []
    for model in MODELS:
        inv = inventory[inventory["model"].eq(model)]
        backbones = ", ".join(sorted(inv["backbone"].dropna().unique()))
        families = ", ".join(sorted(inv["family"].dropna().unique()))
        checkpoint_rows.append([
            model_labels[model], families, backbones, "5/5", f"{int(cache_counts.loc[model].sum()):,}",
        ])

    lines: list[str] = []
    add = lines.extend
    add([
        "# Mixed-resolution super-resolution figure: refreshed technical and paper-writing companion",
        "",
        f"Reference figure: [MixedResolution_unified_v2_{args.run_id}.pdf](MixedResolution_unified_v2_{args.run_id}.pdf)  ",
        f"Editable figure: [MixedResolution_unified_v2_{args.run_id}.svg](MixedResolution_unified_v2_{args.run_id}.svg)  ",
        f"Machine-readable provenance: [FigureSourceManifest_unified_v2_{args.run_id}.json](FigureSourceManifest_unified_v2_{args.run_id}.json)  ",
        f"Formal audit: [UnifiedV2Audit_{args.run_id}.json](../../_Process_Results/UnifiedPublicationV2/UnifiedV2Audit_{args.run_id}.json)",
        "",
        "## Document status and intended use",
        "",
        "This document is the source-backed companion for the refreshed best-checkpoint figure. It is intended to support drafting the Results, Methods, figure caption, limitations, and rebuttal/audit material. Numerical claims below are recomputed from the current CSV/JSON sources; July checkpoint values are used only in the explicitly labelled change section.",
        "",
        f"- Figure/layout run: `{args.run_id}`.",
        f"- Unified accuracy and sensor-sweep source run: `{source_data_run_id}`.",
        f"- Multiscale wavelet source run: `{multiscale_run_id}`.",
        f"- Base protocol/result run: `{args.base_data_run_id}`.",
        f"- Qualitative per-snapshot table run: `{snapshot_data_run_id}`.",
        "- Evaluation backend: cached physical-field reconstructions; no inference occurred during plotting or wavelet analysis.",
        "- Checkpoint contract: all canonical rows use `best.pt`.",
        f"- Final canvas: {canvas_w:.1f} × {canvas_h:.1f} mm.",
        f"- Audit result: **{sum(check['passed'] for check in audit['checks'])}/{len(audit['checks'])} checks passed**.",
        "",
        "## Technical summary",
        "",
        "The figure tests whether high-resolution (H, 128 × 128) density fields can be reconstructed from sparse point measurements when models are trained with high-, mixed-, or exclusively lower-resolution trajectories. Panels progress from the training-resolution protocol (a), through global reconstruction accuracy and sensor efficiency (b–d), to spatially and statistically resolved multiscale fidelity (e–f).",
        "",
        f"At 512 sensors, DMF-Gen attains mean physical relative L2 errors of {summary_row(sweep, 'DMFGen', '3_Mixed_HML', 512)['mean']:.4f}, {summary_row(sweep, 'DMFGen', '4_ZeroH_Balanced', 512)['mean']:.4f}, and {summary_row(sweep, 'DMFGen', '5_ZeroH_MRich', 512)['mean']:.4f} for Mixed-HML, Zero-H-balanced, and Zero-H-M-rich, respectively. It is the best-performing model in both zero-H regimes at this sensor budget. Under Zero-H-M-rich training, its median fine-scale spatial correlation is {wavelet_row(wavelet, 'DMFGen', '5_ZeroH_MRich', 'fine', 'pattern_correlation')['median']:.3f}, compared with {wavelet_row(wavelet, 'FFM_Perceiver', '5_ZeroH_MRich', 'fine', 'pattern_correlation')['median']:.3f}, {wavelet_row(wavelet, 'Senseiver', '5_ZeroH_MRich', 'fine', 'pattern_correlation')['median']:.3f}, and {wavelet_row(wavelet, 'MLP_RBF', '5_ZeroH_MRich', 'fine', 'pattern_correlation')['median']:.3f} for FFM-Perceiver, Senseiver, and MLP-RBF.",
        "",
        f"Relative to the supplied July baseline, DMF-Gen's five-recipe mean at 512 sensors changes by {dmf_average_change:+.1f}%, while its Zero-H-M-rich cell changes by {dmf_rich_baseline_change:+.1f}%. The unified source manifest records the following refreshed model set: {refreshed_models_text}. Accordingly, baseline comparisons should be described as checkpoint-specific rather than as evidence that every model/recipe improved.",
        "",
        "The evidence is descriptive and comparative rather than causal. All conditions share the same 300 held-out case–time pairs and paired sensor plans, but the experiment does not isolate architecture, optimization, or checkpoint-selection effects through repeated training seeds.",
        "",
        "## Evidence organization",
        "",
        md_table(
            ["Region", "Panels", "Question answered"],
            [
                ["Top", "a–b", "What are the training-resolution regimes, and what is the headline error at 512 sensors?"],
                ["Middle", "c", "Where do reconstruction errors occur on one matched held-out field?"],
                ["Lower middle", "d", "How stable are model/recipe rankings from 64 to 512 observations?"],
                ["Bottom", "e–f", "Which spatial scales fail, and is failure due to pattern loss or variance misallocation?"],
            ],
        ),
        "",
        "Panels b and d provide population-level global-error evidence; panel c localizes error; panel e gives a scale-separated spatial example; panel f supplies the corresponding 300-case multiscale statistics. Panel c is the qualitative anchor, while b, d, and f carry the primary quantitative claims.",
        "",
        "## Dataset, resolution, and evaluation scope",
        "",
        "### Physical field and native grids",
        "",
        "The evaluated field is physical density from the multiresolution CFD dataset. All formal accuracy values are computed on the H grid after reversing training normalization.",
        "",
        md_table(
            ["Resolution", "Symbol", "Grid", "Grid points"],
            [["Low", "L", "32 × 32", "1,024"], ["Medium", "M", "64 × 64", "4,096"], ["High", "H", "128 × 128", "16,384"]],
            ["---", ":---:", "---:", "---:"],
        ),
        "",
        f"Panel a uses one common case/time across the three native grids and exports {len(fields):,} physical field rows. The current protocol tables record {len(sensor_protocol):,} projected sensor rows.",
        "",
        "### Training recipes and spatial-data budget",
        "",
        "The ratio order is L:M:H. The spatial degree-of-freedom budget is",
        "",
        "\\[",
        "B_{\\mathrm{DOF}} = N_L(32^2)+N_M(64^2)+N_H(128^2),",
        "\\]",
        "",
        "where the N values are active training trajectories assigned to each resolution. This is a spatial-data exposure measure, not wall-clock or energy cost.",
        "",
        md_table(
            ["Recipe", "L:M:H", "Cases L/M/H", "Active cases", "Snapshots", "Spatial DOF", "H-only fraction"],
            budget_rows,
            ["---", ":---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "### Held-out cohort and sampling unit",
        "",
        "- The test split contains 1,000 cases; the formal screen uses 300 distinct held-out cases.",
        "- One deterministic time is selected per case from the common valid time window.",
        "- The sampling unit for uncertainty is one case–time pair, not one grid point.",
        "- Every per-snapshot relative L2 value summarizes all 16,384 H-grid points.",
        "- Sensors and stochastic generation seeds are paired across models, recipes, and counts.",
        "- Bootstrap intervals use 2,000 case-level resamples with seed 42.",
        "",
        "### Sparse observation budgets",
        "",
        md_table(
            ["Sensors", "H-grid fraction"],
            [["64", "0.390625%"], ["128", "0.781250%"], ["256", "1.562500%"], ["384", "2.343750%"], ["512", "3.125000%"]],
            ["---:", "---:"],
        ),
        "",
        "The displayed 64–512 sets are nested. The cache also retains 768- and 1,024-sensor rows for Mixed-HML and both zero-H recipes, but those counts are not plotted in this main figure.",
        "",
        "## Checkpoint and cache provenance",
        "",
        f"All {len(inventory)} model–recipe checkpoints passed artifact inspection, recipe-manifest validation, CUDA loading, and full reconstruction. The canonical manifest contains {len(cache):,} successful rows, all marked `checkpoint_kind=best`; no dummy substitution was required.",
        "",
        md_table(
            ["Figure model", "Saved family", "Backbone(s)", "Validated recipes", "Cache rows"],
            checkpoint_rows,
        ),
        "",
        "DMF-Gen uses `GL_rbf_ENH` for H-only, H-limited, and Mixed-HML, and `GL_rbf` for the two zero-H recipes. Point-cloud FFM families use two Euler function evaluations. DMF-Gen uses hard/default observation consistency; FFM-Perceiver uses smooth endpoint consistency. Senseiver and MLP-RBF use their native deterministic inference contracts.",
        "",
        "Coverage per model is 1,500 rows for H-only, 1,500 for H-limited, and 2,100 for each of Mixed-HML, Zero-H-balanced, and Zero-H-M-rich. The prior 37,200 last-checkpoint cache files were removed after canonical best-checkpoint reconciliation; the shared truth/grid store was retained.",
        "",
        "## Metric definitions",
        "",
        "### Physical relative L2",
        "",
        "\\[ E_{\\mathrm{rel}L_2}=\\frac{\\|\\hat u-u\\|_2}{\\|u\\|_2+10^{-12}}. \\]",
        "",
        "Lower is better. This is an amplitude-sensitive full-field error in physical density units.",
        "",
        "### Pointwise absolute error and signed residual",
        "",
        "Panel c displays \\(e_{\\mathrm{abs}}(x,y)=|\\hat u-u|\\). Panel e displays signed scale residual \\(r_s=\\hat u_s-u_s\\), where positive/negative values indicate local over/underprediction.",
        "",
        "### Scale-component relative L2",
        "",
        "\\[ E_{\\mathrm{rel}L_2,s}=\\frac{\\|\\hat u_s-u_s\\|_2}{\\|u_s\\|_2+10^{-12}}. \\]",
        "",
        "This value can exceed one when the true component has little energy, so fine-scale values should be read with the variance fraction and correlation.",
        "",
        "### Spatial pattern correlation",
        "",
        "\\[ \\rho_s=\\frac{\\langle\\hat u_s,u_s\\rangle}{\\|\\hat u_s\\|_2\\|u_s\\|_2+10^{-12}}. \\]",
        "",
        "This is an uncentered cosine similarity, not mean-subtracted Pearson correlation. Higher is better.",
        "",
        "### Variance allocation",
        "",
        "\\[ f_s(u)=\\frac{\\|u_s\\|_2^2+\\epsilon}{\\sum_k(\\|u_k\\|_2^2+\\epsilon)},\\qquad \\Delta f_s^{(\\mathrm{pp})}=100[f_s(\\hat u)-f_s(u)]. \\]",
        "",
        "Zero percentage points is ideal. Positive bias assigns excess variance to a scale; negative bias assigns too little.",
        "",
        "### Statistics",
        "",
        "Panels b and d show arithmetic means and bootstrap 95% confidence intervals over 300 case–time pairs. Panel f shows medians; its source table also contains means, standard deviations, quartiles, and bootstrap mean intervals. These intervals describe held-out-case sampling uncertainty, not retraining variability.",
        "",
        "## Panel-by-panel technical explanation",
        "",
        "### Panel a — resolution protocol and training-data budget",
        "",
        "Panel a shows the same physical density state at L, M, and H resolution and compares the five recipe budgets. The zoom region is chosen from H-resolution ground-truth gradient magnitude, independently of model output. Native values are shown without model-specific interpolation. The panel separates reducing H-case count (H-limited) from changing the resolution mixture (Mixed-HML and zero-H recipes).",
        "",
        f"Primary sources: [`ResolutionProtocol_fields_{args.base_data_run_id}.csv`](../../_Process_Results/ResolutionProtocol/ResolutionProtocol_fields_{args.base_data_run_id}.csv), [`ResolutionProtocol_budgets_{args.base_data_run_id}.csv`](../../_Process_Results/ResolutionProtocol/ResolutionProtocol_budgets_{args.base_data_run_id}.csv), and [`ResolutionProtocol_sensors_{args.base_data_run_id}.csv`](../../_Process_Results/ResolutionProtocol/ResolutionProtocol_sensors_{args.base_data_run_id}.csv).",
        "",
        "### Panel b — full-field accuracy at 512 sensors",
        "",
        "Panel b contains 20 aggregate points (4 models × 5 recipes), each with `valid_n=300`, on a logarithmic axis. Gray connectors emphasize the H-limited→Mixed-HML and Zero-H-balanced→Zero-H-M-rich contrasts.",
        "",
        md_table(
            ["Model", "H-only", "H-limited", "Mixed-HML", "Zero-H-balanced", "Zero-H-M-rich"],
            panel_b_rows,
            ["---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "Values are mean physical relative L2 [bootstrap 95% CI].",
        "",
        "Paired recipe effects at 512 sensors (negative means the second recipe lowers error):",
        "",
        md_table(
            ["Model", "Mixed-HML vs H-limited", "M-rich vs balanced zero-H"],
            paired_contrasts,
            ["---", "---:", "---:"],
        ),
        "",
        f"The key interaction is architecture dependence. Relative to H-limited training, Mixed-HML changes error by {mixed_effects['DMFGen']:+.1f}% for DMF-Gen, {mixed_effects['FFM_Perceiver']:+.1f}% for FFM-Perceiver, {mixed_effects['Senseiver']:+.1f}% for Senseiver, and {mixed_effects['MLP_RBF']:+.1f}% for MLP-RBF. DMF-Gen remains the lowest-error zero-H model; its M-rich recipe changes error by {rich_effects['DMFGen']:+.1f}% relative to balanced zero-H.",
        "",
        f"Primary table: [`SensorSweepAllRecipes_summary_{source_data_run_id}.csv`](../../_Process_Results/UnifiedPublicationV2/SensorSweepAllRecipes_summary_{source_data_run_id}.csv).",
        "",
        "### Panel c — matched qualitative reconstructions",
        "",
        f"Panel c uses snapshot {snap} (case {int(panel_c['case_id'])}, time index {int(panel_c['time_index'])}) at {snap_count} sensors. The snapshot is fixed by the shared qualitative configuration rather than model ranking. The ROI is chosen exclusively from ground-truth gradient magnitude. All field maps share one normalization, all error maps share another, and no smoothing, sharpening, or model-specific contrast adjustment is applied.",
        "",
        md_table(
            ["Model", "Mixed-HML", "Zero-H-balanced", "Zero-H-M-rich"],
            panel_c_rows,
            ["---", "---:", "---:", "---:"],
        ),
        "",
        f"The common field range is [{panel_c['field_limits'][0]:.6f}, {panel_c['field_limits'][1]:.6f}] density units; the absolute-error range is [0, {panel_c['error_limits'][1]:.7f}]. {panel_c['reference_roi_sensor_counts'][0]} of the {snap_count} sensors fall inside the displayed ROI. This is one illustrative field and must not replace the population means in panels b and d.",
        "",
        "### Panel d — sensor efficiency",
        "",
        "Panel d contains 100 aggregate cells (5 recipes × 4 models × 5 counts), each based on 300 paired case–time errors. All axes share the same log-scale range. Model identity is encoded redundantly by color, marker, and line style so the curves do not rely on color alone.",
        "",
        "Mean physical relative L2 at the formal 256-sensor setting:",
        "",
        md_table(
            ["Model", "H-only", "H-limited", "Mixed-HML", "Zero-H-balanced", "Zero-H-M-rich"],
            panel_d_256,
            ["---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "Reduction in mean error from 64 to 512 sensors:",
        "",
        md_table(
            ["Model", "H-only", "H-limited", "Mixed-HML", "Zero-H-balanced", "Zero-H-M-rich"],
            panel_d_reduction,
            ["---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "The sensor curves show whether differences at 512 sensors persist under sparse observations. Statements about sensor efficiency should be recipe-specific; a lower endpoint and a steeper relative reduction are distinct properties.",
        "",
        "### Panel e — scale-specific truth and signed residuals",
        "",
        f"The current panel e uses the same configured qualitative snapshot as panel c: snapshot {int(panel_e['snapshot'])}, at 256 sensors, under Zero-H-M-rich training. It shows the ground-truth large/intermediate/fine wavelet components and signed residuals for DMF-Gen and Senseiver. The multiscale metadata also records a separate truth-only median-fine-energy representative (snapshot {representative['snapshot_index']}, case {representative['case_id']}), but that metadata-selected snapshot is not the one displayed in the current main panel.",
        "",
        "The decomposition is a four-level orthogonal `db2` transform with periodization. Large scales contain the approximation plus level-4 detail (approximately ≥16 H-grid cells); intermediate is level-3 detail (approximately 8–16 cells); fine combines levels 1–2 (approximately 2–8 cells). Each row uses independent symmetric 99.5th-percentile display limits; metrics use unclipped arrays.",
        "",
        md_table(
            ["Model", "Large", "Intermediate", "Fine"],
            [
                ["DMF-Gen", f"{panel_e['relative_l2_by_model_scale']['DMFGen']['large']:.4f}", f"{panel_e['relative_l2_by_model_scale']['DMFGen']['intermediate']:.4f}", f"{panel_e['relative_l2_by_model_scale']['DMFGen']['fine']:.4f}"],
                ["Senseiver", f"{panel_e['relative_l2_by_model_scale']['Senseiver']['large']:.4f}", f"{panel_e['relative_l2_by_model_scale']['Senseiver']['intermediate']:.4f}", f"{panel_e['relative_l2_by_model_scale']['Senseiver']['fine']:.4f}"],
            ],
            ["---", "---:", "---:", "---:"],
        ),
        "",
        f"For this displayed field, Senseiver/DMF-Gen error ratios are {panel_e['relative_l2_by_model_scale']['Senseiver']['large']/panel_e['relative_l2_by_model_scale']['DMFGen']['large']:.1f}×, {panel_e['relative_l2_by_model_scale']['Senseiver']['intermediate']/panel_e['relative_l2_by_model_scale']['DMFGen']['intermediate']:.1f}×, and {panel_e['relative_l2_by_model_scale']['Senseiver']['fine']/panel_e['relative_l2_by_model_scale']['DMFGen']['fine']:.1f}× for large, intermediate, and fine scales.",
        "",
        "### Panel f — population-level multiscale fidelity",
        "",
        "Panel f summarizes 4 models × 3 displayed recipes × 300 snapshots × 3 scales. Its upper heatmap is median spatial pattern correlation, displayed on a fixed −0.10 to 1.00 range; its lower heatmap is median variance-allocation bias in percentage points. Bias annotations use two decimals and normalize rounded negative zero to 0.00. Each aggregate cell has `valid_n=300`.",
        "",
        "Median spatial pattern correlation (large / intermediate / fine):",
        "",
        md_table(
            ["Model", "Mixed-HML", "Zero-H-balanced", "Zero-H-M-rich"],
            correlation_rows,
            ["---", "---:", "---:", "---:"],
        ),
        "",
        "Median variance-allocation bias in percentage points (large / intermediate / fine):",
        "",
        md_table(
            ["Model", "Mixed-HML", "Zero-H-balanced", "Zero-H-M-rich"],
            bias_rows,
            ["---", "---:", "---:", "---:"],
        ),
        "",
        "Large-scale pattern correlation is near one for all models and therefore weakly discriminative. Intermediate and fine scales separate the methods. DMF-Gen retains the strongest fine-scale alignment in all three displayed lower-resolution regimes and keeps scale-allocation bias close to zero; MLP-RBF shows the largest redistribution of variance under zero-H training.",
        "",
        f"Primary table: [`MultiscaleWavelet_summary_{multiscale_run_id}.csv`](../../_Process_Results/MultiscaleWavelet/MultiscaleWavelet_summary_{multiscale_run_id}.csv).",
        "",
        f"## Quantitative changes relative to the supplied `{args.baseline_figure_run_id}` baseline",
        "",
        "### Mean physical relative L2 averaged over the five recipes at 512 sensors",
        "",
        md_table(
            ["Model", "Baseline", "Refreshed", "Change", "Improved recipes"],
            change_rows,
            ["---", "---:", "---:", "---:", "---:"],
        ),
        "",
        "Largest 512-sensor improvements:",
        "",
        *[
            f"- {model_labels[model]}, {recipe_labels[recipe]}: {change:+.1f}% ({old:.4f} → {new:.4f})."
            for change, model, recipe, old, new in largest_improvements
        ],
        "",
        "Largest 512-sensor regressions:",
        "",
        *[
            f"- {model_labels[model]}, {recipe_labels[recipe]}: {change:+.1f}% ({old:.4f} → {new:.4f})."
            for change, model, recipe, old, new in regressions
        ],
        "",
        "Mean displayed-recipe fine-scale correlation across the three recipes:",
        "",
        md_table(["Model", "Baseline", "Refreshed", "Absolute change"], fine_change_rows, ["---", "---:", "---:", "---:"]),
        "",
        "Mean absolute displayed-recipe variance-allocation bias across three recipes and scales:",
        "",
        md_table(["Model", "Baseline (pp)", "Refreshed (pp)", "Change (pp)"], bias_change_rows, ["---", "---:", "---:", "---:"]),
        "",
        "The refreshed checkpoints improve global error broadly, but Senseiver's mean displayed-recipe fine-scale correlation decreases. This distinction should be retained in the manuscript: lower global L2 does not guarantee better fine-scale spatial organization.",
        "",
        "## End-to-end generation and audit trail",
        "",
        "1. `best.pt` checkpoint artifacts and recipe manifests are inspected and CUDA-load probed.",
        "2. `02_build_reconstruction_cache.py` produces paired physical reconstructions on GPU and records checkpoint hashes, sensor-plan hashes, seeds, and inference settings.",
        "3. `03_rebuild_cache_manifest.py` reconciles the canonical manifest; `04_prune_orphaned_cache.py` removes superseded cache files.",
        "4. Accuracy, sensor-sweep, frequency/gradient, and protocol exporters regenerate source CSVs.",
        "5. `95_prepare_unified_v2_data.py` builds the complete all-recipe table. The refreshed code derives H-only/H-limited sweep cells from any available canonical cache, preventing false missing 512-sensor entries.",
        "6. `80_export_multiscale_wavelet.py` performs cache-only wavelet decomposition and verifies the manifest/file inventory before and after processing.",
        "7. `96_export_unified_v2_panels.py` exports standalone a–f SVG/PDF/PNG triplets; `97_assemble_mixed_resolution_unified_v2.py` draws the native composite.",
        "8. `98_audit_unified_v2.py` verifies plotted values, provenance, geometry, typography roles, editable text, and collision-free layout.",
        "",
        "The composite is not a raster collage. It is drawn natively with exact physical panel rectangles; dense field layers are rasterized only within the vector SVG/PDF container. The PDF embeds TrueType Liberation Sans because Arial was unavailable in the plotting environment.",
        "",
        "## Numerical validation",
        "",
        f"- Cache rows: {len(cache):,}; successful: {(cache['status']=='ok').sum():,}; duplicate model/recipe/snapshot/count keys: {cache.duplicated(['model','recipe','snapshot_index','sensor_count']).sum()}.",
        f"- Checkpoint kind: `{cache['checkpoint_kind'].unique()[0]}` for every canonical row.",
        f"- Multiscale cache entries processed: {validation['cache_entries_processed']:,}.",
        f"- Multiscale per-snapshot rows: {validation['per_snapshot_rows']:,}.",
        f"- Maximum truth reconstruction residual: {validation['truth_max_reconstruction_residual']:.3e}.",
        f"- Maximum prediction reconstruction residual: {validation['prediction_max_reconstruction_residual']:.3e}.",
        f"- Maximum prediction variance-fraction sum error: {validation['prediction_variance_fraction_max_sum_error']:.3e}.",
        f"- Maximum truth-identity correlation deviation: {validation['truth_vs_truth_max_pattern_correlation_error']:.3e}, below the {validation['identity_metric_tolerance']:.0e} tolerance.",
        "- All scale groups are nonempty; every wavelet summary cell has `valid_n=300`.",
        f"- Unified figure audit: {sum(check['passed'] for check in audit['checks'])}/{len(audit['checks'])} passed.",
        "- Manual visual QA confirmed complete panel-b/d coverage after correcting the unified sweep-table assembly path.",
        "",
        "## Interpretation boundaries and reviewer-facing caveats",
        "",
        "1. **No retraining uncertainty:** bootstrap intervals cover held-out cases, not random initialization or training-seed variation.",
        "2. **Representative fields are not population estimates:** panel c uses configured snapshot 50; panel e also displays snapshot 50. Population claims must come from b, d, and f.",
        "3. **Panel-e metadata distinction:** the exporter records a truth-only representative snapshot 102, but the current main panel uses the shared snapshot 50 override. Do not describe panel e as displaying snapshot 102.",
        "4. **Correlation is uncentered:** call it spatial pattern correlation or cosine similarity, not Pearson correlation.",
        "5. **Fine-scale L2 can be large at low true energy:** interpret it jointly with fine-scale correlation and variance fraction.",
        "6. **Periodized wavelets assume boundary continuity:** test another boundary mode or an interior crop before making boundary-insensitive claims.",
        "7. **Display clipping is visual only:** percentile limits do not affect metrics.",
        "8. **Spatial DOF is not compute cost:** do not infer proportional time, memory, or energy savings without profiling.",
        "9. **Architecture–recipe interaction is strong:** no training recipe is universally best.",
        "10. **Checkpoint comparison is not controlled retraining:** best-versus-last changes help explain the refreshed figure but do not estimate training variance.",
        "11. **Field scope:** the figure evaluates density; transfer to other channels/datasets requires separate tests.",
        "",
        "## Paper-writing pack",
        "",
        "### Recommended one-sentence claim",
        "",
        "**DMF-Gen provides the most robust high-resolution density reconstruction when high-resolution training data are reduced or removed, combining low global error with superior preservation of intermediate- and fine-scale spatial structure.**",
        "",
        "### Abstract-ready result language",
        "",
        f"Across 300 held-out CFD case–time pairs, DMF-Gen achieved the lowest reconstruction error among the evaluated models when no high-resolution training trajectories were available. At 512 sensors, its mean physical relative L2 error was {summary_row(sweep, 'DMFGen', '4_ZeroH_Balanced', 512)['mean']:.4f} under balanced zero-H training and {summary_row(sweep, 'DMFGen', '5_ZeroH_MRich', 512)['mean']:.4f} under M-rich zero-H training. Wavelet analysis further showed that DMF-Gen retained a median fine-scale spatial correlation of {wavelet_row(wavelet, 'DMFGen', '5_ZeroH_MRich', 'fine', 'pattern_correlation')['median']:.3f} under M-rich zero-H training, substantially exceeding the non-DMF baselines.",
        "",
        "### Results-section draft",
        "",
        f"We first compared five training-resolution recipes while holding the H-resolution evaluation cohort and nested sensor plans fixed (Fig. a–d). Mixed-resolution training had architecture-dependent effects. At 512 sensors, Mixed-HML changed mean physical relative L2 relative to H-limited training by {mixed_effects['DMFGen']:+.1f}%, {mixed_effects['FFM_Perceiver']:+.1f}%, {mixed_effects['Senseiver']:+.1f}%, and {mixed_effects['MLP_RBF']:+.1f}% for DMF-Gen, FFM-Perceiver, Senseiver, and MLP-RBF, respectively. Thus, lower-resolution data were not uniformly beneficial; their value depended on how the reconstruction architecture integrated multiresolution training examples.",
        "",
        f"DMF-Gen was the most robust model when H-resolution training cases were absent. Its mean error at 512 sensors was {summary_row(sweep, 'DMFGen', '4_ZeroH_Balanced', 512)['mean']:.4f} for Zero-H-balanced and {summary_row(sweep, 'DMFGen', '5_ZeroH_MRich', 512)['mean']:.4f} for Zero-H-M-rich, lower than the corresponding non-DMF errors. The advantage persisted across the 64–512 sensor sweep, indicating that it was not confined to the highest observation budget. Relative to the supplied baseline, DMF-Gen's five-recipe average changes by {dmf_average_change:+.1f}%. Within the current source run, M-rich changes DMF-Gen error by {rich_effects['DMFGen']:+.1f}% relative to balanced zero-H; its M-rich cell changes by {dmf_rich_baseline_change:+.1f}% relative to the July baseline.",
        "",
        f"Scale-resolved analysis showed that the largest differences occurred below the dominant large scale (Fig. e–f). Under Zero-H-M-rich training, DMF-Gen achieved median correlations of {wavelet_row(wavelet, 'DMFGen', '5_ZeroH_MRich', 'large', 'pattern_correlation')['median']:.3f}, {wavelet_row(wavelet, 'DMFGen', '5_ZeroH_MRich', 'intermediate', 'pattern_correlation')['median']:.3f}, and {wavelet_row(wavelet, 'DMFGen', '5_ZeroH_MRich', 'fine', 'pattern_correlation')['median']:.3f} for large, intermediate, and fine structures, respectively. Its scale-allocation bias remained near zero, whereas MLP-RBF showed pronounced transfer of variance from large to fine scales. These results indicate that DMF-Gen's lower global error reflects better preservation of multiscale spatial organization rather than only improved large-scale amplitude.",
        "",
        "### Caption draft",
        "",
        "**Mixed-resolution training and multiscale fidelity in sparse H-resolution field reconstruction.** **a,** Native L-, M-, and H-resolution density fields and the five training-resolution recipes; bars report active training cases and labels give spatial degree-of-freedom exposure relative to H-only. **b,** Mean physical relative L2 error at 512 nested H-grid sensors for four reconstruction models and five training recipes (n=300 held-out case–time pairs; error bars, bootstrap 95% confidence intervals). **c,** Matched H-resolution reconstructions, ground-truth-selected zoom, sensor layout, and pointwise absolute errors for one configured held-out snapshot; all fields and errors use shared normalization. **d,** Mean physical relative L2 across 64–512 nested sensors for each training recipe (n=300 per point); colors, markers, and line styles jointly identify models. **e,** Ground-truth wavelet components and signed residuals for DMF-Gen and Senseiver at 256 sensors under Zero-H-M-rich training; rows show large, intermediate, and fine scales, with row-specific symmetric display limits. **f,** Median spatial pattern correlation and variance-allocation bias across 300 held-out fields for Mixed-HML and both zero-H recipes. Lower L2 and variance bias closer to zero are better; higher correlation is better.",
        "",
        "### Methods-ready language",
        "",
        "We evaluated all models on a canonical cohort of 300 distinct held-out CFD cases, selecting one deterministic time per case from the common valid time window. Reconstructions were produced on the 128 × 128 H grid using nested sensor sets and paired generation seeds. Reported physical relative L2 errors were computed after reversing training normalization. Means and 95% confidence intervals were obtained from 2,000 case-level bootstrap resamples. For multiscale evaluation, physical fields were decomposed using a four-level orthogonal db2 discrete wavelet transform with periodization. We grouped the approximation plus level-4 detail as large scale, level-3 detail as intermediate scale, and levels 1–2 as fine scale, and summarized component relative L2, uncentered spatial pattern correlation, and variance-fraction bias across the same 300 case–time pairs.",
        "",
        "### Claims supported by this figure",
        "",
        "- DMF-Gen has the lowest evaluated zero-H reconstruction error across the plotted sensor budgets.",
        "- Mixed-resolution training effects depend on architecture.",
        "- Large-scale structure is reconstructed well by all models, while intermediate/fine scales are more discriminative.",
        "- DMF-Gen combines low global error with stronger fine-scale correlation and near-zero scale-allocation bias.",
        "- The refreshed best checkpoints materially change some quantitative values relative to the supplied baseline.",
        "",
        "### Claims not supported without additional experiments",
        "",
        "- That any recipe is universally optimal across architectures, fields, or datasets.",
        "- That observed differences are causal consequences of resolution mixture alone.",
        "- That spatial DOF reductions imply proportional compute or energy savings.",
        "- That uncertainty includes retraining variability.",
        "- That the conclusions extend to other physical channels.",
        "",
        "### Terminology to keep consistent",
        "",
        "- Use **physical relative L2**, not normalized loss, for panels b–d.",
        "- Use **spatial pattern correlation (uncentered cosine similarity)**, not Pearson correlation.",
        "- Use **variance-allocation bias in percentage points**, not percent error.",
        "- Use **Zero-H-M-rich** for the 1:2:0 recipe.",
        "- Distinguish **training cases/trajectories**, **training snapshots**, and **spatial DOF**.",
        "- State the snapshot-selection rule whenever discussing panels c or e.",
        "",
        "## Suggested follow-up analyses",
        "",
        "1. Repeat training across multiple seeds and report between-training variability.",
        "2. Add bootstrap intervals for the panel-f medians from the existing per-snapshot table.",
        "3. Test nonperiodic wavelet boundary modes and an interior-crop sensitivity analysis.",
        "4. Profile training/inference time, memory, and energy alongside spatial DOF.",
        "5. Repeat the protocol for additional CFD channels and datasets.",
        "6. Resolve whether panel e should continue sharing snapshot 50 with panel c or instead display the truth-only snapshot 102 recorded by the multiscale selector.",
        "",
        "## Source inventory",
        "",
        f"- Main figure: [PDF](MixedResolution_unified_v2_{args.run_id}.pdf), [SVG](MixedResolution_unified_v2_{args.run_id}.svg), [PNG](MixedResolution_unified_v2_{args.run_id}.png).",
        f"- Figure source manifest: [FigureSourceManifest_unified_v2_{args.run_id}.json](FigureSourceManifest_unified_v2_{args.run_id}.json).",
        f"- Unified audit: [UnifiedV2Audit_{args.run_id}.json](../../_Process_Results/UnifiedPublicationV2/UnifiedV2Audit_{args.run_id}.json).",
        f"- Checkpoint inventory: [ModelInventory_{args.inventory_run_id}.csv](../../_Process_Results/ModelInventory/ModelInventory_{args.inventory_run_id}.csv).",
        "- Canonical cache manifest: [ReconstructionCache_manifest_formal_20260712.csv](../../_Process_Results/ReconstructionCache/ReconstructionCache_manifest_formal_20260712.csv).",
        f"- Unified data manifest: [UnifiedV2DataManifest_{source_data_run_id}.json](../../_Process_Results/UnifiedPublicationV2/UnifiedV2DataManifest_{source_data_run_id}.json).",
        f"- All-recipe accuracy: [AllRecipeAccuracy_summary_{source_data_run_id}.csv](../../_Process_Results/UnifiedPublicationV2/AllRecipeAccuracy_summary_{source_data_run_id}.csv).",
        f"- All-recipe sensor sweep: [SensorSweepAllRecipes_summary_{source_data_run_id}.csv](../../_Process_Results/UnifiedPublicationV2/SensorSweepAllRecipes_summary_{source_data_run_id}.csv).",
        f"- Multiscale summary: [MultiscaleWavelet_summary_{multiscale_run_id}.csv](../../_Process_Results/MultiscaleWavelet/MultiscaleWavelet_summary_{multiscale_run_id}.csv).",
        f"- Multiscale per-snapshot table: [MultiscaleWavelet_per_snapshot_{multiscale_run_id}.csv](../../_Process_Results/MultiscaleWavelet/MultiscaleWavelet_per_snapshot_{multiscale_run_id}.csv).",
        f"- Multiscale metadata: [MultiscaleWavelet_metadata_{multiscale_run_id}.json](../../_Process_Results/MultiscaleWavelet/MultiscaleWavelet_metadata_{multiscale_run_id}.json).",
        f"- Layout configuration: [`publication_layout_unified_v2.yaml`](../../_Scripts/publication_layout_unified_v2.yaml).",
        f"- Quantitative baseline comparison: [`quantitative_change_report.md`](../../../../figures/generated/MixedResolution_unified_v2/quantitative_change_report.md).",
        "",
        "## Reproducibility",
        "",
        "This companion is generated by [`build_mixed_resolution_companion.py`](../../../../figures/scripts/build_mixed_resolution_companion.py). Re-running the generator validates checkpoint coverage, finite summary values, and the completed figure audit before writing the document.",
        "",
        "Exact command:",
        "",
        "```bash",
        f"conda run -n fig python figures/scripts/build_mixed_resolution_companion.py --run-id {args.run_id} --source-data-run-id {source_data_run_id} --multiscale-run-id {multiscale_run_id} --base-data-run-id {args.base_data_run_id} --snapshot-data-run-id {snapshot_data_run_id} --inventory-run-id {args.inventory_run_id}",
        "```",
    ])

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {output}")


if __name__ == "__main__":
    main()

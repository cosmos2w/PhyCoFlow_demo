# Figure 5 validation workflow

This directory contains the lightweight Figure 5 post-processing and visualization pipeline. It reads checkpoints, reconstruction caches, frozen metrics, and raw datasets in place from the turbulent-combustion and super-resolution project trees; it does not copy or regenerate those products.

The `figure5-validation-v2` contract is a compact six-panel, quantitative figure about empirical conditional uncertainty and computational characteristics. The main figure contains no qualitative map and makes no scientific claim from a proxy. Formal field identities are the paper's `Y_CH4`, `Y_CO`, `T`, `U1`, and `p`; under `Cond_T`, the plotted unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p` in that order.

## Panel map

- `a` — empirical calibration of central conditional intervals.
- `b` — sharpness as interval width normalized by training-set field standard deviation.
- `c` — state-level normalized spread associated with ensemble-mean relative-L2 error.
- `d` — actual eight-method native-mesh accuracy–latency comparison.
- `e` — aligned DMF-Gen query-latency and peak-allocated-memory micro-axes.
- `f` — DMF-Gen latency–error path annotated by measured vector-field evaluation count.

The composed target is 183 mm × approximately 145 mm and SVG-only. Spatial uncertainty examples, raw physical-unit widths, extended uncertainty diagnostics, detailed cost tables, and ablations belong in the SI or a later figure.

## Layout

- `configs/` — in-place source roots, formal panel contract, exact field/method order, palette, and build policy.
- `scripts/` — command-line build and QA entry points.
- `utils/` — reusable data adapters, statistics, styling, and panel renderers.
- `figures/generated/` — timestamped SVG outputs; ignored by git.
- `docs/` — figure contract and timestamped panel companions.
- `results/` — lightweight derived CSVs and build manifests; ignored by git.

## Build

Use the project figure environment:

```bash
conda run -n fig python Dis_SI_Process/scripts/build_figure5_draft.py --strict-formal
```

To reproduce an exact name:

```bash
conda run -n fig python Dis_SI_Process/scripts/build_figure5_draft.py \
  --strict-formal --timestamp YYYYMMDD_HHMM
```

The V2 output contract is six standalone SVG panels, one composed SVG, a Markdown provenance companion for every output, lightweight derived tables, and a JSON build manifest. PDF is unsupported during this testing stage.

## Formal-data handoff

Freeze the U1/U2 uncertainty products and native-method/DMF cost products below the `ValidationV2` roots named in `configs/figure5_draft.yaml`. Reuse the frozen 1,000-state FieldL2 result in place only when its checkpoint identity matches exactly. Stream new UQ calculations state-by-state; do not retain full ensembles except for predeclared SI cases.

`--strict-formal` is the manuscript-candidate mode. It must fail if any required panel source, identity, cohort/protocol metadata, or required statistic is missing, and it must never substitute a proxy or pending panel. Legacy engineering inputs remain listed only to preserve the existing QA/provenance path; they cannot enter the strict-formal composition or support manuscript claims.

The frozen inference specification is `_TrainedModels/_ValidationPlans/validation_v1.yaml`. Run model inference in `phycoflow_env` (the `fig` environment is plotting-only):

```bash
conda run -n phycoflow_env python \
  0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Scripts/60_run_uncertainty_validation.py \
  --config 0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_ValidationPlans/validation_v1.yaml \
  --job PILOT --device cuda:0

conda run -n phycoflow_env python \
  0_demo_TurbulentCombustion/tools/benchmark_validation_v2.py \
  --plan 0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_ValidationPlans/validation_v1.yaml \
  --suite all --methods all --device cuda:0
```

Run the uncertainty pilot before U1/U2/U3. The production commands differ only in `--job U1`, `--job U2`, or `--job U3`. Do not lower the frozen state/draw counts for a formal run.

## QA

```bash
conda run -n fig python Dis_SI_Process/scripts/qa_figure5_outputs.py \
  Dis_SI_Process/figures/generated/<timestamp>
```

QA should check the six V2 filenames, SVG parseability, editable text, 183 mm × approximately 145 mm composition, paper-aligned field/method identities, source companions, and absence of proxy/pending content in a strict-formal build. Visual inspection at final printed size remains required.

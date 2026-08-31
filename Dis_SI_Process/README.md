# Figure 5 V3 validation workflow

This directory refactors the Figure 5 V2 post-processing workflow into the five-panel V3 comparison while leaving every V2 result and companion untouched. It reads checkpoints, frozen Figure 4 FieldL2 tables, the adopted sensor plan, and reusable V2 DMF UQ summaries in place; it does not duplicate checkpoints, datasets, reconstruction caches, or raw ensemble stacks.

The V3 main figure compares conditional ensemble quality across the five trained generative methods and measures clean computational trade-offs across all eight Figure 4 methods. Under `Cond_T`, `Y_CH4`, `Y_CO`, `U1`, and `p` are equal-weight macro-aggregated.

## Panel map

- `a` — normalized empirical CRPS for DMF-Gen, FFM-FNO, FFM-Perceiver, Latent FM, and SiT.
- `b` — method-wise Spearman association between macro normalized ensemble spread and macro ensemble-mean error.
- `c` — corrected native N=40,300 accuracy–latency trade-off for all eight methods.
- `d` — warm model-core latency versus N, with curves only for native variable-query paths.
- `e` — peak allocated memory versus N under the identical support protocol.

Full reliability/interval-width curves, fieldwise UQ, diversity, cold/no-cache timing, reserved memory, and NFE/solver diagnostics remain SI/internal. No ablation training is part of this workflow.

## Execution order

Use `phycoflow_env` for inference and benchmarking. GPU 2 is referenced explicitly below.

```bash
conda run -n phycoflow_env python \
  0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Scripts/61_run_uq_compare_v3.py \
  --plan 0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_ValidationPlans/validation_v1.yaml \
  --job PILOT --methods all --device cuda:2 --run-id <pilot_run_id>
```

Only after the 12×8 pilot passes stochasticity and same-seed reproducibility:

```bash
conda run -n phycoflow_env python \
  0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Scripts/61_run_uq_compare_v3.py \
  --plan 0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_ValidationPlans/validation_v1.yaml \
  --job FORMAL --methods all --device cuda:2 --run-id uq_compare_formal_20260830_v3r6
```

Run clean cost only when GPU 2 has no foreign compute process and no UQ job is active:

```bash
conda run -n phycoflow_env python \
  0_demo_TurbulentCombustion/tools/benchmark_validation_v3.py \
  --plan 0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_ValidationPlans/validation_v1.yaml \
  --device cuda:2 --run-id formal_cost_clean_v3_20260830_v3
```

The cost runner enforces at least 20 warmups, 30 synchronized repeats, and 10 measured seconds per accepted row. DMF uses its canonical configured 8,192-point reconstruction chunk; the timing audit maps the provisional approximately 29 ms probe to the earlier 4,096-point streaming boundary. The runner saves clean-GPU state, timing-boundary audit, exact identity checks, native results, variable-query support, latency repeats/summaries, allocated/reserved memory, and QA below `results/ValidationV3/CostClean/`.

## Strict figure build

Use the plotting-only `fig` environment:

```bash
conda run -n fig python Dis_SI_Process/scripts/build_figure5_draft.py --strict-formal
```

The output is five timestamped standalone SVGs, one 183 mm × 118 mm composed V3 SVG, one Markdown companion per panel plus a composed companion, V3 source tables/build manifest, and a timestamped completion report. `--strict-formal` fails if the five-method UQ or clean V3 cost run is missing, failed, identity-incomplete, or replaced by V2/proxy data.

## QA

```bash
conda run -n fig python Dis_SI_Process/scripts/qa_figure5_outputs.py \
  Dis_SI_Process/figures/generated/<timestamp> --strict-formal
```

QA checks six exact SVG names, parseability, editable text, fixed composed dimensions, required V3 terminology, companion/source-table presence, support-key consistency, absence of V2 main-panel content, and strict build-manifest status. Visual inspection at final printed size remains mandatory.

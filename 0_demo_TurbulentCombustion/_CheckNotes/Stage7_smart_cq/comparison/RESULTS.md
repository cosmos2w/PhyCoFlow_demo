# Stage 7 reference comparison: Senseiver and latent FM

## Scope and comparability

This closes the compact reference comparison requested after Stage-7 model
selection. No baseline source or checkpoint was modified. Senseiver and latent
FM use their archived `Cond_T/last.pt` checkpoints at epoch 5000; Stage7-All256
uses `epoch_1000.pt` with EMA weights.

The cost benchmark is matched where the architectures permit it:

- RTX 6000 Ada GPU 1;
- batch 128, 4096 supervised/training query points, and 256 T sensors;
- median of three measurements after one warmup;
- full 40,300-point reconstruction for inference;
- model already loaded, with sparse-condition construction and decoding
  included but checkpoint/dataset loading and CPU result copying excluded.

The baseline trainers historically use `CH4/CO/T/U1/p`, while Stage 7 uses
`CO/T/U0/U1/p`. Therefore the archived reconstruction results are contextual
reference evidence, not a strict five-field quality ranking against Stage 7.

## Matched cost

| Model | Executable / trainable parameters | B128/Q4096 step | projected compute epoch† | training peak | full-grid inference | inference peak | NFE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Senseiver | 8.340M / 8.340M | **213.06 ms** | **15.13 s** | 15,327.6 MiB | **16.23 ms** | 601.1 MiB | 1 |
| Latent FM | 88.540M / 81.014M | 356.80 ms | 25.33 s | **9,330.1 MiB** | 31.01 ms | 1,639.0 MiB | 4 |
| **Stage7-All256** | **5.491M / 5.491M** | 397.06 ms | 28.19 s | 20,239.5 MiB | 33.11 ms | **374.3 MiB** | 4 |

† Projection is 71 compute steps for 9,000 training snapshots and excludes
data-loader wait, logging, validation, and checkpoint I/O. Stage 7 uses its
formal exact-gradient 2048-query execution microbatch; the baselines use their
native trainer steps. Senseiver's archived training config is B256, but it was
remeasured at B128 here for the requested comparison.

Additional inference points:

- Stage7-All256 NFE1: 19.14 ms at 40,300 points.
- Latent FM NFE2: 24.09 ms at 40,300 points.
- Stage7-All256 also has the separately validated persistent 1M-query NFE4
  result of 285.7 ms with geometry built once and `static_features` caching.

## Reconstruction evidence

| Model/evidence set | Field set and sample scope | Mean relative L2 | common unobserved CO/U1/p mean |
|---|---|---:|---:|
| Senseiver archived Cond-T | CH4/CO/T/U1/p, 1,000 test snapshots | 0.142990 unobserved-field mean | **0.150352** |
| Latent FM archived Cond-T, NFE2 | CH4/CO/T/U1/p, 1,000 test snapshots | 0.453104 unobserved-field mean | 0.507183 |
| Stage7-All256 NFE1 diagnostic | CO/T/U0/U1/p, one fixed validation snapshot | **0.213053 five-field mean** | 0.276836 |
| Stage7-All256 NFE4 diagnostic | CO/T/U0/U1/p, one fixed validation snapshot | 0.234270 five-field mean | 0.308279 |

The first two rows come from the frozen 1,000-snapshot paper archive and the
last two from the Stage-7 fixed-snapshot reconstruction. Different field sets,
sample counts, and sensor manifests prevent a defensible direct ranking from
these absolute values. The controlled Stage-7 quality decision remains the
fixed-manifest RF and matched F0/CQ reconstruction analysis in
`../evaluation_1000/RESULTS.md`.

## Structural interpretation

- **Senseiver** is deterministic, supervised, and one-pass. Its lower step and
  inference cost are expected because it predicts one conditional estimate and
  performs no RF trajectory. It remains a strong deterministic reference, but
  it is not a replacement for generative full-function-space RF.
- **Latent FM** evolves a learned compressed latent field. Its compact spatial
  grid explains the low training peak and competitive NFE4 full-grid latency,
  despite 16.1x more executable parameters than Stage7-All256. Its VAE and
  latent velocity network solve a structurally different problem.
- **Stage7-All256** performs full-function-space generative RF while retaining
  the compact 128-D repeated query decoder. It has the fewest parameters and
  lowest full-grid inference peak here, and its persistent pointwise path is the
  only one validated at 1M queries in this study.

## Recommendation

The comparison does not change the Stage-7 decision: use Stage7-All256 as the
default CQ configuration, CQ-LR-128 as the throughput-first CQ option, and
Senseiver/latent FM as structurally distinct deterministic/compressed-latent
references rather than substitutable implementations.

## Evidence

- `archived_baseline_benchmark.json`
- `stage7_selected_inference.json`
- `compact_comparison.csv`
- `benchmark_archived_baselines.py`
- `benchmark_selected_stage7_inference.py`
- archived quality source:
  `Save_TrainedModel/_TrainedModels/_Process_Results/FieldL2/FieldL2_summary_paper_full_20260711.csv`


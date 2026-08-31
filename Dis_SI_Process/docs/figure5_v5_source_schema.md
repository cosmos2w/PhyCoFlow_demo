# Figure 5 V5 source and statistical contract

## Figure contract

- Core conclusion: DMF-Gen has the strongest measured conditional-distribution quality; empirical ensemble uncertainty is tested separately at state and spatial levels; accepted training and inference measurements expose the adopted checkpoint's lifecycle footprint without claiming matched-budget causal efficiency.
- Evidence archetype: focused `2 × 2` quantitative validation grid.
- Backend/export: Python/Matplotlib in the `fig` environment; 183 mm × 116 mm composed canvas; SVG only; editable text; low-alpha state/bootstrap clouds may be embedded-rasterized inside the SVG to control file complexity.
- Dataset/task: turbulent-combustion missing-channel reconstruction, `Cond_T`, M=256, native N=40,300.
- Unobserved fields: `Y_CH4`, `Y_CO`, `U1`, `p`, macro-aggregated with equal 0.25 weights.
- Method order a–c: DMF-Gen, FFM-FNO, FFM-Perceiver, Latent FM, SiT.
- Method order d: the five generators followed by MLP-RBF, Geo-FNO and Senseiver.

## Reuse matrix and panel map

### a — Probabilistic reconstruction

Read formal V3 `uq_compare_formal_20260830_v3r6/per_state_method.csv` and `crps_summary.csv` in place. The plot contains 200 paired state CRPS values per method, the formal mean, and the unchanged temporal moving-block-bootstrap 95% CI. No ensemble or bootstrap computation is rerun.

### b — Uncertainty tracks difficult states

Read the same formal V3 state table in place and deterministically reproduce the exact 2,000-replicate moving-block-bootstrap distribution using the original SHA-256 seed rule, block length 25 and method salt. The open marker and horizontal interval are the formal full-sample Spearman association and its adopted CI. This is uncertainty informativeness, not calibration or prospective error prediction.

### c — Uncertainty localizes reconstruction error

Run `run_error_capture_v5.py` only because no all-state/all-method pointwise ensemble summaries were retained. For each method and state, hold 64 matching-seed draws in memory, compute physical ensemble mean and sample s.d., reduce each unobserved field immediately to the frozen spatial fractions `[0.05,0.10,0.20,0.30,0.40,0.50,0.75,1.0]`, and discard the stack before the next state. The state macro curve is the equal mean of four independently normalized field curves. Summary bands use the same temporal moving-block bootstrap as panels a/b. `EC-AUC` integrates `C(q)-q` after adding the exact origin `(0,0)`.

Retained files under `results/ValidationV5/UQLocalization/<run_id>/` are only:

- `error_capture_curves.csv` — one compact state/method row with macro and field-resolved curve ordinates;
- `error_capture_summary.csv` — macro and fieldwise state means, temporal 95% CIs and EC-AUC;
- `manifest.json` and `qa.json`.

No full ensemble fields, per-draw files, scratch arrays or bootstrap arrays are retained.

### d — Training–inference lifecycle footprint

Read accepted V3 clean native inference latency and frozen Figure-4 error in place. Convert formal canonical update timings using

\[
T_{GPUh}=\sum_s t^{update}_sN^{update}_sG_s/(3.6\times10^6).
\]

The metric is named exactly **Replay-equivalent model-core training GPU-hours**. Adopted update counts come from exact stage/run metadata. Latent FM is the sum of both required sequential stages. Geo-FNO uses synchronized max-rank wall time from the formal 2-GPU DDP global-batch-192 replay and `G=2`. Historical wall time, file timestamps, validation, checkpointing and data I/O are excluded.

Retained files under `results/ValidationV5/Lifecycle/<run_id>/` are only `lifecycle_summary.csv`, `lifecycle_stage_provenance.csv`, `manifest.json` and `qa.json`.

## Dataset-aware display schema

Every row in the timestamped `figure5_v5_source.csv` includes at least:

```text
dataset,task,condition,panel,method,checkpoint_sha256,
state_id,cohort_id,metric_name,metric_value
```

The table contains only exact plotted state/bootstrap samples, summaries, curve coordinates and lifecycle coordinates; it is not a copy of any checkpoint, raw inference bundle or old result directory.

## Supplementary Information routing

The timestamped figure directory contains an `si/` subdirectory with SVG-only plots for calibration/interval width, fieldwise CRPS and spread/error association, fieldwise error-capture curves, 40.3k–8M query latency/memory stress and NFE diagnostics. These are not duplicated in the main 2×2 figure.

## Strict-formal and review-risk gates

- a/b require the exact five methods, paired 200-state cohort, S=64 and original manifest/QA identities.
- c requires genuine stochastic ensembles, exact matching protocol, state-first field macro aggregation, monotone bounded curves ending at one, and temporal bootstrap bands.
- d requires accepted native timings, all adopted update counts, every required stage, correct GPU count, exact checkpoint identity and no historical-wall-time claim.
- The lifecycle view is descriptive and hardware/configuration qualified.
- Formal reliability evidence remains underdispersed, so V5 uses “empirical conditional ensemble uncertainty” and makes no perfect-calibration or Bayesian-posterior claim.

# Stage 6 formal current-architecture baseline results

## Decision

**Retain F0 as the formal baseline. Do not replicate F1 with additional seeds.**

At the single matched training seed, F1 provides no material convergence or
reconstruction improvement. Its best checkpoint is slightly worse under the
controlled fixed-manifest RF evaluation and under matched best-checkpoint field
reconstruction, while requiring substantially more time and GPU memory.

This decision does not claim that larger supervision is universally harmful. It
means the present evidence does not justify paying its cost or promoting it over
F0 before Stage 6 decoder changes.

## Completion and protocol audit

- Both formal histories contain epochs 1–200 with final validation and cached
  Euler NFE 1/2/4 reconstruction.
- Final launcher logs contain no traceback, OOM, or training error.
- Both final runs use seed 42, batch size 64, identical architecture/optimizer/
  split/RF settings, and the same optimized Stage 1–5 data path.
- F0 uses 4,096 monolithic queries. F1 uses 16,384 effective queries, 8,192-query
  microbatches, and reused condition context.
- Batch size was reduced equally after F1 could not fit a full step at batch 144
  or 96 on a 48 GiB GPU. Failed startups are not included in the results.

## Convergence

| Metric | F0 | F1 | Interpretation |
|---|---:|---:|---|
| Best validation RF loss | 0.520273 (epoch 180) | 0.521483 (epoch 200) | F1 is 0.23% worse |
| Final validation RF loss | 0.523435 | 0.521483 | F1 is 0.37% better at this single point |
| Mean training loss, epochs 181–200 | 0.521463 | 0.523079 | F1 is 0.31% worse |

The curves closely track throughout training. F1's small final-validation
advantage does not survive best-checkpoint selection or controlled evaluation.

## Controlled fixed-manifest evaluation

Both best checkpoints were evaluated on the same 64 validation layouts with
three controlled RF draws per layout (192 evaluations per checkpoint). RF draws
are technical repeats; they were averaged within each layout before calculating
the layout-level paired interval.

| Metric | F0 best | F1 best |
|---|---:|---:|
| Checkpoint epoch | 180 | 200 |
| Mean RF loss | 0.475357 | 0.476982 |

F1 minus F0 is `+0.001625`, or **+0.342%** relative to F0. The layout-level 95%
CI is `[-0.006065, +0.009315]`; paired t-test `p=0.674`. The interval crosses
zero and the training-seed sample size is one, so this is evidence of no clear
benefit—not proof of exact equivalence.

## Matched best-checkpoint reconstruction

The reconstruction comparison uses the same validation snapshot, 256 T sensors,
sensor checksum, RF sample seed, Euler solver, and cache/streaming path.

| Mean of five field-relative L2 errors | F0 best | F1 best | F1 vs F0 |
|---|---:|---:|---:|
| NFE 1 | 0.295012 | 0.301421 | +2.17% |
| NFE 2 | 0.325034 | 0.332732 | +2.37% |
| NFE 4 | 0.360421 | 0.366576 | +1.71% |

F1 is worse for the five-field mean at every tested NFE. The largest consistent
degradation is in `U_1`; the only clear per-field improvement is CO at NFE 1
(`-0.00439` absolute relative-L2 difference).

## Efficiency cost

| Cost metric | F0 | F1 | F1 / F0 |
|---|---:|---:|---:|
| Mean training time/epoch, epochs 2–200 | 26.93 s | 95.82 s | 3.56x |
| Total training time | 1.50 h | 5.32 h | 3.56x |
| Diagnostic training-step time | 267.8 ms | 654.7 ms | 2.44x |
| Sampled peak reserved GPU memory | 18,198 MB | 33,634 MB | 1.85x |

## Stage 6 baseline

Use the F0 best checkpoint at epoch 180 as the current-architecture reference:

`F0_frozen_current_DemoN9300_20260821_075633/best.pt`

Judge Stage 6 query-decoder candidates against F0 using the same fixed manifest,
RF seeds, reconstruction snapshot/sensors/sample seed, and explicit cost metrics.
Only promote a decoder that improves reconstruction or convergence materially
without hiding the runtime/memory tradeoff.

## Artifacts

- Machine-readable summary and source tables:
  `figures/generated/stage6_formal_baseline/`
- Controlled fixed-manifest results:
  `_CheckNotes/Stage6_formal_baseline/evaluation/fixed_manifest_best.json/.csv`
- Matched reconstruction metrics and local arrays:
  `_CheckNotes/Stage6_formal_baseline/evaluation/matched_reconstruction/`
- Reproducible plotting script:
  `figures/scripts/plot_stage6_formal_baseline.py`
- Decision figure:
  `stage6_formal_baseline_decision.svg/.pdf/.png/.tiff`
- Matched field plate:
  `stage6_matched_reconstruction_fields.svg/.pdf/.png/.tiff`

## Limitations

- There is one training seed per protocol; no training-seed uncertainty can be
  estimated.
- Fixed-manifest RF repeats quantify layout/RF behavior, not training variability.
- The matched spatial plate uses one validation snapshot.
- F0 and F1 best checkpoints occur at different epochs (180 and 200).

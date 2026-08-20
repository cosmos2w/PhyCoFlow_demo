# Round 1: 100-Epoch Legacy vs Optimized Data-Path Comparison

## Status

Completed on 2026-08-20. Both matched 100-epoch runs exited successfully on
physical GPU 1. Each loss history has exactly 100 epochs, both final validation
passes completed, the timing and GPU telemetry files are complete, and the
post-run analyzer passed its audit.

## Comparison question

Does the optimized point-cloud FFM data path reduce training computation and wall
time without degrading 100-epoch convergence when the model, RF objective,
dataset, seed, optimizer, batch/query/observation sizes, and evaluation schedule
are held fixed?

## Controlled setup

| Dimension | Legacy run | Optimized run |
|---|---|---|
| Physical GPU | 1 | 1 |
| GPU model | RTX 6000 Ada Generation | RTX 6000 Ada Generation |
| Epochs | 100 | 100 |
| Seed | 42 | 42 |
| Dataset | `Merged_COTU0U1P.h5` | Same |
| Train split | 90%, 9,000 snapshots | Same |
| Batch size | 144 | 144 |
| Steps per epoch | 63 | 63 |
| Query points | 4,096 | 4,096 |
| Observations | Uniformly sampled 192–384 for field 1 | Same count distribution |
| Backbone | `GL_rbf_ENH` | Same |
| Gather | `topk_rbf_glres`, K=32, KeOps | Same |
| Optimizer | AdamW, LR `1e-4`, WD `1e-6` | Same |
| Prior/objective | RFF prior, unchanged RF loss | Same |
| Validation | Epochs 1 and 100 only | Same |
| Reconstruction | Disabled (`save_every=1000`) | Disabled |
| Data diagnostics | Disabled | Disabled |
| Loss PNG plotting | Disabled | Disabled |
| Tqdm scalar interval | Every 20 steps | Every 20 steps |
| Normalization statistics | Same precomputed five-channel stats file | Same |

The identical logging interval deliberately removes the historical every-step
tqdm synchronization from this experiment. The comparison therefore targets the
core data-path differences, not terminal-rendering overhead.

## Data-path variables under test

| Component | Legacy | Optimized |
|---|---|---|
| Coordinate batching | Per-item full coordinate clones and batch stacking | One shared fixed mesh |
| Index sampler | Full GPU `randperm(N_full)` | Scalable CPU unique sampler |
| Observation layout | Constructed after full H2D transfer | Sampled before field materialization |
| Field read | Full snapshot | Full snapshot (matched because indexed HDF5 was already shown pathological) |
| GPU transfer | Full `[B,N_full,C]` fields and coordinates | Selected query/observation tensors only |
| DataLoader workers | Non-persistent historical behavior | Persistent, prefetch factor 2 |
| H2D transfer | Blocking historical behavior | Nonblocking from pinned host tensors |

## Measurement method

The runs are launched sequentially by `_CheckNotes/run_round1_compare.sh` only
after GPU 1 is free. Instrumentation intentionally stays outside the hot model
path:

* `perf_counter` records train, validation, and total epoch seconds in the normal
  loss-history CSV/JSON (three scalar clock reads per phase, no CUDA sync).
* `/usr/bin/time -v` records end-to-end elapsed time, CPU time, and maximum host
  resident memory for each process.
* `nvidia-smi` samples utilization, device memory, and board power every two
  seconds into a separate telemetry file for each run.
* No data-path diagnostic CUDA events, profiler, reconstruction, or per-epoch
  figure rendering is active.

Both runs use the same physical GPU, eliminating inter-device variation and
cross-job GPU contention. The fixed legacy-then-optimized order can warm the OS
filesystem cache for the second run, so end-to-end I/O-sensitive wall-time gains
must be interpreted with that order effect in mind. Remaining variation can also
come from different stochastic index sequences (legacy samples on GPU while the
optimized path samples on CPU), asynchronous kernels, and shared-system noise.

## Results

The optimized data path reduced steady training time by **11.26%**, increased
sample throughput by **12.68%**, and reduced end-to-end wall time by **11.19%**.
Its final training loss differed from legacy by only **+0.03%**, and its mean
loss over the final ten epochs differed by **+0.05%**. The single final
validation estimate was **2.05% higher**, so this run supports the efficiency
change without showing a material training-convergence regression, but it does
not establish validation equivalence statistically.

| Primary metric | Legacy | Optimized | Optimized vs legacy |
|---|---:|---:|---:|
| External wall time | 3,043.47 s | 2,703.00 s | -11.19% (1.126x speedup) |
| Steady train time/epoch, epochs 6–100 | 30.253 s | 26.848 s | -11.26% |
| Train time/step | 0.4802 s | 0.4262 s | -11.26% |
| Training throughput | 297.49 samples/s | 335.23 samples/s | +12.68% |
| Final train loss | 0.664788 | 0.664987 | +0.03% |
| Final-10 mean train loss | 0.665923 | 0.666241 | +0.05% |
| Epoch-100 validation loss | 0.660833 | 0.674357 | +2.05% |

## Computation-cost comparison

| Metric | Legacy | Optimized | Change |
|---|---:|---:|---:|
| Total measured train-phase time | 3,024.67 s | 2,686.67 s | -11.17% |
| Steady median train time/epoch | 30.247 s | 26.862 s | -11.19% |
| Steady p95 train time/epoch | 30.397 s | 26.935 s | -11.39% |
| User + system CPU time | 6,443.18 s | 4,171.62 s | -35.25% |
| Maximum host RSS | 3.906 GiB | 1.677 GiB | -57.06% |
| Peak GPU memory | 41,595 MiB | 41,489 MiB | -0.25% |
| Mean GPU utilization | 93.66% | 97.91% | +4.25 percentage points |
| Mean board power | 268.46 W | 287.38 W | +7.05% |
| Estimated GPU energy | 227.00 Wh | 215.85 Wh | -4.91% |

The large host-memory and CPU-time reductions are consistent with removing
batch-stacked full coordinates and avoiding full-field GPU materialization. Peak
GPU memory changes little in full training because model activations dominate at
this batch/query size; the earlier isolated data-path benchmark showed the
selected input tensors themselves use substantially less GPU memory. Higher GPU
utilization indicates that the optimized loader keeps the unchanged model fed
more consistently. Although mean power rises while the GPU is busier, shorter
runtime lowers estimated total GPU energy by about 11.15 Wh.

## Wall-clock comparison

The optimized run completed **340.47 seconds (5 min 40 s) sooner**, reducing
end-to-end time from 50 min 43 s to 45 min 03 s. The internal train-phase saving
was 338.00 seconds, so almost all of the external saving came from the repeated
training loop rather than startup, validation, or shutdown.

The fixed run order is a limitation: legacy ran first and populated the OS page
cache. `/usr/bin/time` consequently recorded 15,809,136 filesystem-input units
for legacy and only 3,336 for optimized. This cannot explain the complete steady
epoch difference because epochs 6–100 exclude startup/warm-up and the dataset had
already been traversed repeatedly, but the 11.19% end-to-end result should still
be treated as an observed same-GPU result rather than a cache-neutral causal
estimate. A reversed-order repetition would quantify that residual order effect.

## Convergence comparison

| Epoch | Legacy train loss | Optimized train loss | Optimized difference |
|---:|---:|---:|---:|
| 1 | 1.832628 | 1.856244 | +1.29% |
| 5 | 1.220943 | 1.230053 | +0.75% |
| 10 | 0.998555 | 1.028254 | +2.97% |
| 25 | 0.791650 | 0.793046 | +0.18% |
| 50 | 0.717439 | 0.704419 | -1.81% |
| 75 | 0.680729 | 0.683380 | +0.39% |
| 100 | 0.664788 | 0.664987 | +0.03% |

Both runs reached 75% of their own initial loss at epoch 3 and 50% at epoch 15.
Because each optimized epoch is faster, those milestones arrived in 81.91 s and
403.70 s, versus 89.72 s and 454.18 s for legacy. The best training losses were
0.661394 at legacy epoch 95 and 0.662348 at optimized epoch 98.

Validation loss was nearly identical at epoch 1 (1.572894 legacy versus 1.571174
optimized) and 2.05% higher for optimized at epoch 100. Query and observation
sampling are stochastic and occur on different RNG/device streams in the two
profiles, so these runs intentionally do not share exact per-step sample layouts.
The dedicated identical-index tests establish tensor-materialization equivalence;
multiple training seeds or fixed pre-generated layouts would be needed to decide
whether the final validation difference is systematic.

## Interpretation and recommendation

Adopt the optimized data path as the preferred training configuration for this
model, while retaining the explicit legacy profile for the planned component
ablations. It delivers a repeatable-looking steady-state speed improvement,
substantially lowers host memory/CPU cost, and preserves the 100-epoch training
trajectory without changing GL-RBF, top-k gather, RF loss, prior, or model
architecture.

For the next formal benchmark, run at least three seeds and alternate run order
on the same GPU. Keep `field_read_mode=legacy_full_snapshot` for this contiguous
HDF5 file: the earlier isolated benchmark found `indexed_union` pathologically
slower. The next performance round can then focus separately on reconstruction
streaming/caching and model-side query cost, which were deliberately excluded
here.

## Validation audit

* Branch: `perf/pointcloud-ffm-field-reconstruction`.
* Both jobs exited 0; each history contains one header plus 100 epoch rows.
* The only config differences are the named data-path components plus run ID and
  output directory; device, dataset, model, optimization, seed, and schedule are
  identical.
* Full regression suite on physical GPU 1: **17 passed in 6.62 s**.
* `py_compile`, launcher shell syntax, analyzer audit, and `git diff --check`
  passed.
* `src/Model.py` and `src/direct_coherence_loss.py` have no diff. No GL-RBF,
  top-k/KeOps, RF objective, prior, or reconstruction mathematics changed.

## Artifacts

* Legacy config: `_CheckNotes/config_round1_legacy_100.yaml`
* Optimized config: `_CheckNotes/config_round1_optimized_100.yaml`
* Launcher: `_CheckNotes/run_round1_compare.sh`
* Runtime logs/telemetry: `_CheckNotes/Round1_runtime/`
* Machine-readable analysis: `_CheckNotes/Round1_runtime/analysis.json`
* Run directories: `_CheckNotes/Round1_runs/legacy/` and
  `_CheckNotes/Round1_runs/optimized/`

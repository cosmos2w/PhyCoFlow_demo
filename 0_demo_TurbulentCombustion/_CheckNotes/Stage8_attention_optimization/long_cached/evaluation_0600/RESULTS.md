# Stage8 cached/full long-run evaluation through epoch 600

Status: complete. The training process was intentionally stopped after the
complete epoch-650 validation. `epoch_0600.pt` is the formal matched milestone;
`last.pt` preserves epoch 650 and `best.pt` preserves the online-validation best
at epoch 625.

Checkpoint SHA256 values:

- epoch 600: `3d30a1adbc198f3fd95ba761012f8e97c7809315ee3105229e508ac29a7367bb`
- online best, epoch 625: `1e6746b476a0ecb81f5551ec7f2d0aa6e028f698521023ec05f8d3fefd8a5c81`
- last, epoch 650: `262e3ee010e8952becc81351ead2a69b9a5bf35f84d8ffb22cec27a0dbd939a1`

## Compared executions

Both candidates use the Stage7-All256 scientific model and protocol: latent
width 256, 128 latents, four blocks, CQ-128 low-rank/additive decoding,
sinusoidal FiLM, measurement/support features, Top-K 32, KeOps, B128/Q4096,
2048-query microbatches, cosine horizon 1000, and EMA 0.999. The intended model
execution difference is:

| candidate | condition attention | sensor padding | K/V projections |
|---|---|---|---:|
| Stage7-All256 | `legacy_mha` | full | 4 |
| Stage8-Cached | `cached_kv` | full | 1 |

The Stage7 run was resumed after epoch 200, so it is a matched historical
protocol rather than a bitwise paired uninterrupted training control. Frozen
checkpoint correctness and the paired Stage8 smoke provide the strict
execution-equivalence gates.

## Fixed-manifest RF

Protocol: the same 64 frozen validation layouts x three controlled RF repeats,
RF seed 1729, batch size 1, and EMA weights. Cached-minus-Stage7 confidence
intervals are paired over all 192 rows.

| epoch | Stage7 RF | cached RF | cached change | paired difference 95% CI |
|---:|---:|---:|---:|---:|
| 200 | 0.407096 | 0.419612 | +3.07% | [0.00870, 0.01633] |
| 400 | 0.309678 | 0.311324 | +0.53% | [-0.00095, 0.00425] |
| 600 | 0.276424 | 0.276226 | -0.07% | [-0.00255, 0.00215] |

Cached training is measurably behind at epoch 200, but the gap closes by epoch
400. At epoch 600 the RF means differ by only `0.000199` (0.07%), and the paired
interval comfortably includes zero. There is no mature-training RF degradation
detectable under this protocol.

## Matched reconstruction

All rows share validation snapshot 0, the exact 256-temperature-sensor layout,
observation seed 42, RF seed 1729, EMA weights, Euler integration, and persistent
static-feature reconstruction. Lower is better.

| epoch | NFE | Stage7 mean rel-L2 | cached mean rel-L2 | cached change |
|---:|---:|---:|---:|---:|
| 200 | 1 | 0.286619 | 0.284913 | -0.60% |
| 200 | 4 | 0.331069 | 0.322426 | -2.61% |
| 400 | 1 | 0.241133 | 0.239568 | -0.65% |
| 400 | 4 | 0.266632 | 0.271949 | +1.99% |
| 600 | 1 | 0.221609 | 0.216705 | -2.21% |
| 600 | 4 | 0.247025 | 0.242908 | -1.67% |

At epoch 600 the worst field remains `U_1`, but cached is lower at both NFE1
(`0.45758` versus `0.49523`) and NFE4 (`0.55575` versus `0.58567`). This is one
matched reconstruction snapshot and should be interpreted as a deterministic
diagnostic, not a population confidence interval.

## Epoch time and memory

The long-run timing window uses `train_seconds` for epochs 250–600, after the
Stage7 resume boundary and warm-up. It contains 351 epochs per candidate.

| metric | Stage7 legacy/full | cached/full | change |
|---|---:|---:|---:|
| mean train time/epoch | 27.362 s | 25.689 s | **6.11% faster** |
| median train time/epoch | 27.362 s | 25.690 s | **6.11% faster** |
| controlled B128/Q4096 step | 363.164 ms | 339.993 ms | **6.38% faster** |
| controlled peak allocated | 20,761.6 MiB | 20,261.4 MiB | **2.41% lower** |

The independent long-run and controlled-step measurements agree. The memory
comparison comes from the frozen shared-checkpoint benchmark so trained-weight
differences cannot confound it.

## EMA behavior

Both candidates contain all 148 EMA shadow tensors, use decay 0.999, validate
and select best checkpoints with EMA, and have exactly matched update counts.

| epoch | updates, both | Stage7 EMA/live rel-L2 | cached EMA/live rel-L2 |
|---:|---:|---:|---:|
| 200 | 14,200 | 0.013954 | 0.014005 |
| 400 | 28,400 | 0.007867 | 0.007997 |
| 600 | 42,600 | 0.004096 | 0.004109 |

The decay behavior is effectively the same, and the EMA/live distance contracts
smoothly for both runs. Cached execution did not disrupt EMA updates,
checkpointing, or evaluation selection.

## Decision

Stopping at epoch 650 is justified. The epoch-600 formal checkpoint shows no RF
accuracy damage, matched reconstruction is slightly better overall, EMA behavior
is normal, and the efficiency improvement is reproduced over 351 real epochs.

Promote `cached_kv + full` as the public/default GL_rbf_CQ execution while
retaining `legacy_mha + full` as a compatibility/debug mode. It does **not**
clear the original Stage8 target of at least 8% whole-step speedup: measured
gains are approximately 6.1–6.4%, with 2.4% lower peak allocation. The final
selection explicitly accepts that smaller, independently reproduced efficiency
gain because the long validation closes the early RF gap and shows no mature-
quality damage.

The epoch-600 cached checkpoint is validation evidence, not a replacement for
the higher-quality Stage7 epoch-1000 release weights. A later code release can
use cached/full execution with the existing portable Stage7 EMA-resolved
checkpoint because the parameter schema is unchanged and strict checkpoint
compatibility has already passed.

## Evidence

- `fixed_manifest.json` and `fixed_manifest.csv`
- `matched_reconstruction/summary.json` and per-candidate field metrics
- `rf_comparison.csv`
- `reconstruction_comparison.csv`
- `ema_comparison.csv`
- `comparison.json`
- Stage8 controlled benchmark: `../../benchmark.json`

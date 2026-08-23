# Stage 7 Results

Status: correctness and efficiency gates passed; both 200-epoch screens completed; S7-B selected and resumed toward epoch 1000 on physical GPU 1.

## Correctness

- Focused Stage-7 tests: **11 passed**.
- Existing CQ/cache/microbatch regression groups: **84 passed, 1 skipped**.
- Complete regression suite after implementation: **141 passed, 1 skipped**.
- Frozen clean CQ-LR-128 `best.pt`: strict load succeeded with **0 missing / 0 unexpected** keys.

## Pre-training efficiency gates

Definitive artifact: `benchmarks/pretraining_cost.json` (CSV companion included). GPU 1 was idle with about 48.5 GiB free at benchmark start. Stage-7 formal training uses the proven exact-gradient 2048-query execution microbatch with one shared condition context; the effective scientific batch/query remains B128/Q4096. F0 and frozen CQ-LR remain monolithic.

| Candidate | B128/Q4096 step (ms) | speedup vs F0 | peak allocated (MiB) | reduction vs F0 | 1M/NFE4 (s) | speedup vs F0 |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 544.84 | 1.00x | 27,346 | 0.0% | 0.4367 | 1.00x |
| Frozen CQ-LR-128 | 437.81 | 1.24x | 22,973 | 16.0% | 0.2433 | 1.79x |
| S7-A / Cond128 | 332.27 | 1.64x | 18,624 | 31.9% | 0.3031 | 1.44x |
| S7-B / All256 | 397.06 | 1.37x | 20,239 | 26.0% | 0.2857 | 1.53x |

Both Stage-7 candidates pass all launch thresholds: at least 1.10x training speed, at least 10% allocated-memory reduction, and at least 1.15x persistent 1M/NFE4 speed versus F0. Persistent inference uses the same 32768-query chunk for every candidate and keeps zero post-build KNN calls.

Batch-1 model scaling (full train step, milliseconds):

| Candidate | 4k | 16k | 65k |
|---|---:|---:|---:|
| F0 | 36.90 | 70.43 | 209.88 |
| Frozen CQ-LR-128 | 29.11 | 118.87 | 131.19 |
| S7-A / Cond128 | 43.60 | 38.72 | 77.70 |
| S7-B / All256 | 39.65 | 73.68 | 91.21 |

The short scaling samples are noisier than the formal B128 measurement; they are retained as raw evidence rather than used as the launch gate.

## 200-epoch screen launch

Launched concurrently on physical GPU 1 on 2026-08-22:

| Screen | PID (launcher) | Run directory |
|---|---:|---|
| S7-A | 1529557 | `screen_200/runs/S7_A_Cond128_200ep_B128_DemoN9701_20260822_224828` |
| S7-B | 1529661 | `screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830` |

Logs:

- `screen_200/logs/S7_A_20260822_224755.log`
- `screen_200/logs/S7_B_20260822_224755.log`

Both runs completed 200 epochs without OOM. Concurrent GPU usage stabilized near 46.0 GiB on the 48.5-GiB device.

## Epoch-200 quality decision

The controlled fixed-manifest evaluation uses 192 paired rows per checkpoint
(64 validation layouts x three repeats, RF seed 1729). Full evidence and the
deterministic reconstruction table are in `screen_200/RESULTS.md`.

| Candidate | Epoch-200 RF loss | paired improvement vs F0 | NFE1 mean | NFE4 mean |
|---|---:|---:|---:|---:|
| F0-128 | 0.50517 | 0.0% | 0.2915 | 0.3271 |
| CQ-LR-128 | 0.51974 | -2.9% | 0.3136 | 0.3622 |
| CQ-LR-256 | 0.44633 | +11.6% | 0.2925 | **0.3248** |
| S7-A / Cond128 | 0.49926 | +1.2% | 0.3053 | 0.3485 |
| **S7-B / All256** | **0.40710** | **+19.4%** | **0.2866** | 0.3311 |

S7-B is 18.5% better than S7-A, 8.8% better than CQ-LR-256, and
19.4% better than F0 in controlled epoch-200 RF loss. It also has the best
NFE1 reconstruction mean and remains within 1.2% of F0 in NFE4 mean error.
Its formal cost remains 1.37x faster training, 26.0% lower allocated memory,
and 1.53x faster persistent 1M/NFE4 inference than F0.

Decision: **continue S7-B only**. S7-A and S7-B are not essentially tied.

## Epoch-1000 continuation

`configs/S7_B_All256_1000ep_resume.yaml` resumed the same timestamped S7-B
run from epoch 200 on physical GPU 1 on 2026-08-23. The scheduler and EMA
states resumed in place; pre-resume artifacts are preserved under
`screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830/bk/`.

The final epoch-1000 default-CQ recommendation and the separate MHA-mask vs
SDPA/fused-AdamW kernel study remain pending completion of this one run.

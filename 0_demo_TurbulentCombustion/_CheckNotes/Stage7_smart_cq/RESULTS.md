# Stage 7 Results

Status: correctness and efficiency gates passed; S7-A and S7-B 200-epoch screens launched concurrently on physical GPU 1.

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

Both runs passed model/data initialization and multiple optimizer steps without OOM. Concurrent GPU usage stabilized near 46.0 GiB on the 48.5-GiB device.

## Pending decision

At epoch 200, compare F0, clean CQ-LR-128, existing CQ-LR-256, S7-A, and S7-B with the fixed manifest, matched reconstruction evaluation, worst field, training cost, memory, and persistent 1M/NFE4 inference. Continue at most the best Stage-7 candidate to epoch 1000 unless the two are essentially tied.

No scientific quality/default recommendation is made before the matched epoch-200 evidence exists.

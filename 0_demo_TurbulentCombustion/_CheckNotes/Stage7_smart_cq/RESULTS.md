# Stage 7 Results

Status: complete. S7-B / Stage7-All256 finished epoch 1000, passed final quality and efficiency evaluation, and is recommended as the new default CQ configuration.

## Correctness

- Focused Stage-7 tests after the final EMA frozen-state correction: **12 passed**.
- Existing CQ/cache/microbatch regression groups: **84 passed, 1 skipped**.
- Final complete regression suite: **142 passed, 1 skipped**.
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

The run completed epoch 1000. The exact milestone has train loss 0.29000 and
stored validation loss 0.30290. The stored best checkpoint is epoch 965
(`val_loss=0.29227`), but exact epoch 1000 is marginally better in the controlled
fixed-manifest RF evaluation (0.261507 vs 0.261564) and is the recommended
scientific checkpoint.

## Final controlled recommendation

| Candidate | checkpoint | fixed RF | change vs F0 | NFE1 mean | NFE4 mean |
|---|---:|---:|---:|---:|---:|
| F0 | e1000 / recon best e845 | 0.325531 | baseline | 0.239674 | 0.262493 |
| CQ-LR-128 | e1000 / recon best e845 | 0.357043 | 9.7% worse | 0.264192 | 0.294014 |
| CQ-LR-256† | best e840 | **0.261010** | 19.8% better | **0.210119** | **0.227882** |
| **Stage7-All256** | **e1000** | **0.261507** | **19.7% better** | **0.213053** | **0.234270** |

† CQ-LR-256 stopped at epoch 842 and is retained as an incomplete reference.
Stage7-All256 essentially ties its RF quality (0.19% difference) while completing
the clean 1000-epoch protocol. Versus F0 it improves deterministic reconstruction
mean by 11.1% at NFE1 and 10.8% at NFE4. The first measured Stage7 milestone to
beat final F0 RF quality is epoch 400.

The final audit corrected EMA semantics: only trainable parameters are averaged;
frozen parameters and buffers are copied exactly. Existing Stage-7 checkpoints
are repaired during loading using their live frozen state. All final
reconstruction candidates share the same RF-prior checksum, and the correction
changes S7's fixed RF mean by only 1.2e-6.

**Recommendation: make Stage7-All256 the default balanced CQ configuration.**
Keep CQ-LR-128 as the throughput-first option. F0 is dominated here because
Stage7-All256 is both better quality and 1.37x faster training, uses 26.0% less
allocated GPU memory, and is 1.53x faster for persistent 1M/NFE4 inference.

## Separate kernel result

Parameter/forward/loss/gradient parity passed for MHA-mask versus explicit SDPA.
At B128/Q4096, MHA+unfused AdamW is 403.85 ms and explicit SDPA+unfused AdamW
is 406.66 ms (0.7% slower). Keep the current MHA path; PyTorch 2.5 already
dispatches `need_weights=False` efficiently.

Fused AdamW independently passes one-step parity (maximum parameter delta
1.49e-8) and reduces the MHA full step to 396.55 ms, a modest 1.8% gain. Treat
it as an optional kernel optimization, separate from the scientific default.

Full final evidence is in `evaluation_1000/RESULTS.md`; the publication-ready
Pareto figure is in `figures/generated/stage7_final_pareto/`.

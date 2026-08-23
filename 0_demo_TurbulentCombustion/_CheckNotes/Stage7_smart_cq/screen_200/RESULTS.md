# Stage-7 epoch-200 screen results

Decision: **continue S7-B / Stage7-All256 only**. S7-A and S7-B are not tied.

## Controlled RF quality

All rows use the same 64-sample validation manifest, batch 1, three repeats,
and RF seed 1729 (192 paired evaluations). Stage-7 checkpoints use EMA weights.

- Manifest checksum: `392806184e0257f95f8d7a550ef1fb9ca85a1bd7fa8537d5471807763f1a0822`
- Materialized-input checksum: `b116f09b4ffc40bf821155f86df5a993cacfb0e70ba83830d0727cfd143c0e95`

| Candidate | Epoch-200 RF loss | Paired improvement vs F0 | 95% CI |
|---|---:|---:|---:|
| F0-128 | 0.50517 | 0.0% | reference |
| CQ-LR-128 | 0.51974 | -2.9% | [-5.0%, -0.8%] |
| CQ-LR-256 | 0.44633 | +11.6% | [+8.5%, +14.8%] |
| S7-A / Cond128 | 0.49926 | +1.2% | [-1.2%, +3.6%] |
| **S7-B / All256** | **0.40710** | **+19.4%** | **[+16.7%, +22.1%]** |

S7-B is 18.5% better than S7-A, 8.8% better than the clean CQ-LR-256
control, and 19.4% better than F0 at the same epoch.

## Deterministic reconstruction

All five checkpoints use snapshot 0, the same 256 temperature sensors, hard
observation consistency, and the same RF seed. The shared sparse-condition
checksum is `34a28c7c91bf36425ed86d22927ac03d1bb2053d8d50d71024ee0c84ec570eb3`.
The worst field is `U_1` for every candidate.

| Candidate | NFE1 mean | NFE1 worst | NFE4 mean | NFE4 worst |
|---|---:|---:|---:|---:|
| F0-128 | 0.2915 | 0.7084 | 0.3271 | 0.8440 |
| CQ-LR-128 | 0.3136 | 0.7825 | 0.3622 | 0.9841 |
| CQ-LR-256 | 0.2925 | **0.6855** | **0.3248** | **0.8236** |
| S7-A / Cond128 | 0.3053 | 0.7560 | 0.3485 | 0.9267 |
| **S7-B / All256** | **0.2866** | 0.6859 | 0.3311 | 0.8588 |

S7-B has the best NFE1 mean and effectively ties CQ-LR-256 on the NFE1
worst field. At NFE4 it is 1.2% worse than F0 in mean error and 1.8% worse on
the worst field, while remaining better than S7-A and CQ-LR-128.

## Quality–efficiency decision

| Candidate | Step (ms) | Speedup | Peak allocated | Reduction | 1M/NFE4 | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| F0-128 | 544.84 | 1.00x | 27,346 MiB | 0.0% | 0.4367 s | 1.00x |
| CQ-LR-128 | 437.81 | 1.24x | 22,973 MiB | 16.0% | 0.2433 s | 1.79x |
| CQ-LR-256† | 477.12 | 1.14x | 23,734 MiB | 9.1% | n/a | n/a |
| S7-A / Cond128 | 332.27 | 1.64x | 18,624 MiB | 31.9% | 0.3031 s | 1.44x |
| **S7-B / All256** | **397.06** | **1.37x** | **20,239 MiB** | **26.0%** | **0.2857 s** | **1.53x** |

† CQ-LR-256 uses cumulative training diagnostics rather than the formal Stage-7
benchmark and therefore is shown as an open marker in the Pareto figure.

S7-B clears every formal gate while delivering the strongest epoch-200 RF
quality. S7-A is faster but does not show a statistically clear RF improvement
over F0 and is worse in reconstruction. Only S7-B is continued.

## Continuation

`S7_B_All256_1000ep_resume.yaml` resumed the same run from `last.pt` at epoch
201 on physical GPU 1 on 2026-08-23. Scheduler state resumed at epoch 200,
EMA resumed after 14,200 updates, and the original epoch-200 artifacts were
backed up under `bk/resume_20260823_083814/`.

## Evidence

- `evaluation/comparison_summary.json` and `comparison_table.csv`
- `evaluation/convergence.csv`
- `evaluation/S7_A_fixed_manifest.json` and CSV companion
- `evaluation/S7_B_fixed_manifest.json` and CSV companion
- `evaluation/CQ_LR_L256_fixed_manifest.json` and CSV companion
- `evaluation/matched_reconstruction/summary.json`
- `../../../figures/generated/stage7_epoch200_pareto/`

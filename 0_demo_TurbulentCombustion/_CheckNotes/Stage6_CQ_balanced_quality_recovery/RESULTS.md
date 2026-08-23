# CQ-Balanced quality-recovery result

**Decision: stop before training.** Both the 192-D primary and the sole 224-D
fallback fail the mandatory efficiency gate under the clean batch-128,
4,096-query protocol. No 200-epoch or kernel run was launched.

## Decisive clean-protocol gate

| Candidate | Step (ms) | Speedup vs F0 | Allocated reduction | Reserved reduction | Gate |
|---|---:|---:|---:|---:|---|
| F0 (primary run) | 546.802 | 1.000x | 0.00% | 0.00% | reference |
| CQ-Balanced-192-Full | 538.733 | 1.015x | 4.17% | 0.11% | **FAIL** |
| F0 (fallback run) | 547.997 | 1.000x | 0.00% | 0.00% | reference |
| CQ-Balanced-224-Full | 560.958 | 0.977x | 2.06% | -4.08% | **FAIL** |

Required: at least 1.15x step speedup and at least 10% reduction in allocated
or reserved memory.

## Batch-1 scaling diagnostic

| Queries | Model | Step (ms) | Speedup vs F0 | Allocated reduction | Reserved reduction |
|---:|---|---:|---:|---:|---:|
| 4,096 | F0 | 23.170 | 1.000x | 0.00% | 0.00% |
| 4,096 | CQ-LR | 21.123 | 1.097x | 11.21% | 5.26% |
| 4,096 | CQ-Balanced-192-Full | 25.144 | 0.922x | -1.70% | -8.55% |
| 16,384 | F0 | 51.410 | 1.000x | 0.00% | 0.00% |
| 16,384 | CQ-LR | 37.691 | 1.364x | 28.95% | 29.54% |
| 16,384 | CQ-Balanced-192-Full | 47.752 | 1.077x | 8.15% | 13.35% |
| 65,536 | F0 | 148.328 | 1.000x | 0.00% | 0.00% |
| 65,536 | CQ-LR | 103.171 | 1.438x | 31.53% | 36.83% |
| 65,536 | CQ-Balanced-192-Full | 144.751 | 1.025x | 10.50% | 11.42% |

## Persistent 1M-query inference

Euler NFE=4, persistent geometry plus `static_features`, three repeats:

| Model | Steady latency (s) | Speedup vs F0 | Geometry build (s) | Geometry storage (MiB) |
|---|---:|---:|---:|---:|
| F0 | 0.4250 | 1.000x | 0.1522 | 396.7 |
| CQ-LR | 0.3347 | 1.270x | 0.1467 | 396.7 |
| CQ-Balanced-192-Full | 0.3593 | 1.183x | 0.1444 | 396.7 |


The persistent geometry-only Top-K path remains functional and unchanged. The
192-D candidate is faster than F0 for persistent inference, but that does not
override the failed training efficiency gate.

## Quality/throughput Pareto recommendation

| Model | Clean best validation | Gap vs F0 | B128/Q4096 step | Speedup vs F0 | Peak allocated | Status |
|---|---:|---:|---:|---:|---:|---|
| F0 | 0.353095 | reference | 546.8 ms | 1.000x | 27326.5 MiB | **quality model** |
| CQ-LR | 0.388921 | +10.15% | 445.4 ms | 1.228x | 22953.0 MiB | **throughput model** |
| CQ-Balanced-192 | not measured | n/a | 538.7 ms | 1.015x | 26187.2 MiB | rejected before training |

- Quality model: F0.
- Throughput model: CQ-LR, with its known approximately 10% validation penalty.
- Formal 3-D scientific model: F0; use CQ-LR only for throughput-limited or
  exploratory 3-D work where the validated quality loss is acceptable.
- CQ-Balanced: do not promote; it restores much of F0's cost without clearing
  the pre-training efficiency gate, so spending training time would violate the
  staged protocol.

## Scientific outcome

The structured-concat hypothesis was not quality-screened because its intended
information separation restores too much of F0's query-side cost. CQ-Balanced
therefore has no validation/reconstruction point and cannot enter the quality
Pareto frontier.

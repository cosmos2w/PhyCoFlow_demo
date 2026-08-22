# CQ-LR persistent Top-K — 200-epoch paired result

**PASS: persistent Top-K improves repeated reconstruction without a measurable quality change.**

The paired validator confirms byte-identical training-driver, data-helper, and
data-path source across revisions. Persistent Top-K is inference-only, so training
speed is a neutrality/control measurement; the expected efficiency gain is in
repeated cached-streamed reconstruction on fixed geometry.

| Training/quality metric | No persistent Top-K | Persistent implementation | Change |
|---|---:|---:|---:|
| Mean epoch time, epochs 2–200 (s) | 18.451 | 18.376 | -0.40% |
| Diagnostic step time (ms) | 379.771 | 377.638 | -0.56% |
| Peak allocated (MiB) | 23258.3 | 23258.3 | +0.00% |
| Final validation RF loss | 0.640920 | 0.641611 | +0.11% |
| Epoch-200 fixed-manifest RF loss | 0.567434 | 0.569679 | +0.40% |

| 1M-query Euler NFE-4 reconstruction | Prior Stage-4 static cache | Persistent geometry + static cache |
|---|---:|---:|
| Steady latency (s) | 0.4775 | 0.3579 |
| Speedup | 1.00x | **1.33x** |
| Peak allocated (MiB) | 3410.4 | 3410.4 |
| Top-K searches after cache construction | n/a | 0.0 |

One-time geometry construction costs 0.1458 s and
stores 396.7 MiB at one million queries. The maximum
output difference across persistent benchmark cases is
1.907e-06.

Final checkpoint tensors are not bitwise identical
(maximum parameter difference 2.079e-01).
Full raw metrics and acceptance checks are in `comparison.json`.

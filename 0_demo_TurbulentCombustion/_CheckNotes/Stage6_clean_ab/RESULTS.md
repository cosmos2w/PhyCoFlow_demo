# Clean F0-ENH versus CQ-LR comparison

Both runs use the same extended batch-128 protocol and differ only in run identity and backbone.

| Metric | F0-ENH | CQ-LR | CQ-LR change |
|---|---:|---:|---:|
| Mean epoch time, epochs 2–1000 (s) | 27.150 | 18.294 | -32.62% |
| Diagnostic train step (ms) | 581.410 | 379.276 | -34.77% |
| Peak allocated (MiB) | 27642.9 | 23258.3 | -15.86% |
| Peak reserved (MiB) | 36414.0 | 27688.0 | -23.96% |
| Final validation loss | 0.361207 | 0.400808 | +10.96% |
| Best validation loss | 0.353095 | 0.388921 | +10.15% |

Fixed-manifest results, milestone histories, and threshold times are stored in
`comparison.json`.

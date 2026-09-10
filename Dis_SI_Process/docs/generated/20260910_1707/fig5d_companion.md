# Figure 5 V7R2 panel d: Ablation error distributions

Scientific intent: Show the distribution of unobserved-field relative-L2 across 1,000 held-out snapshots for the five stochastic variants.

Quantitative definition: Statewise relative-L2 values loaded from reconstruction_states.csv and summarized by reconstruction_summary.csv; mean and block-20 interval are source values.

Evidence package: Package A, new ablation source reductions; last.pt only.

Visual design: Horizontal light jitter points plus compact box summary and mean marker; mean numbers only and no percentage annotations.

The state distribution is the five-method last.pt reduction represented by Table S1; all 1,000 points are retained, boxes show Q25--Q75 with median and 1.5-IQR whiskers, and the visible numeric annotation is the mean only with its block-20 interval.

Source ledger entries from `source_manifest.json`:

| key | path | sha256 | role |
| --- | --- | --- | --- |
| reconstruction_states.csv | Dis_SI_Process/results/derived/20260910_1707/reconstruction_states.csv | 535e300ed9b52f61503f265153e56ba78a9cbcda1128d069c4d6515397fbd7fe | Package A V7R2 saved-checkpoint reduction |
| reconstruction_summary.csv | Dis_SI_Process/results/derived/20260910_1707/reconstruction_summary.csv | f6478058231d28d2e48184bb3d0363ac1448bd7c4fa0adf105bf5836152b180a | Package A V7R2 saved-checkpoint reduction |


Output contract: `fig5d_v7r2_20260910_1707.svg` and `fig5d_v7r2_20260910_1707.png`.

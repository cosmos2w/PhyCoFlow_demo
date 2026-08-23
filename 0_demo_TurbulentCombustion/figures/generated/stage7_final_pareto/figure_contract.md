# Figure contract — Stage-7 final quality/throughput Pareto

- Core conclusion: Stage7-All256 improves controlled RF and reconstruction
  quality over F0 while retaining a materially faster, lower-memory CQ path,
  making it the recommended default CQ configuration.
- Figure archetype: quantitative 2x2 decision grid.
- Target/output: double-column technical/publication figure, 183 mm x 140 mm.
- Backend: Python/matplotlib exclusively in the `fig` environment.
- Panel map: a, controlled fixed-manifest convergence; b, paired RF differences
  against exact F0 epoch 1000 with 95% confidence intervals; c, deterministic
  NFE1/NFE4 reconstruction mean and worst-field errors; d, formal
  quality/training-step Pareto with peak memory and persistent inference.
- Evidence hierarchy: panel b is the primary statistical result; panel d is the
  deployment decision view; panels a and c validate convergence and rollout
  behavior.
- Statistics: fixed validation manifest, 192 paired evaluations per checkpoint
  (64 layouts x three repeats), paired mean differences and normal 95% CIs.
- Source data: `evaluation_1000/final_comparison.csv`,
  `evaluation_1000/paired_statistics.csv`, and
  `evaluation_1000/convergence.csv`.
- Image integrity: no raster scientific images or selective image adjustment;
  all panels are plotted directly from tabular evidence.
- Reviewer risks: reconstruction is one fixed validation snapshot, so it is a
  deterministic diagnostic rather than a dataset-wide uncertainty estimate.
  CQ-LR-256 is an incomplete epoch-840 reference and is shown with open/dashed
  styling; it is excluded from the formally matched cost Pareto.

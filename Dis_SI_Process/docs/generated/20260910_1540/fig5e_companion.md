# Figure5e quantitative companion

What does IID’s small global-error improvement conceal in fine-scale velocity?

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Metric and aggregation: Two distinct per-state high-band quantities: canonical trapezoidal shell-mean power ratio (ideal1) and complex-mode residual relative-L2 (ideal0, phase-sensitive). Mean block20 intervals; full/IID only, last policy.

Result and boundary: IID has excess high-band velocity power and roughly twice the high-band residual. Full power is attenuated and is not ideal. This is an index-space, one-draw, two-step diagnostic.

Coordinates at source precision:

| display_label | metric | mean | block20_ci95_low | block20_ci95_high | annotation |
| --- | --- | --- | --- | --- | --- |
| Full model | canonical_shellmean_high_energy_ratio | 0.42853620860075847 | 0.41199518450000661 | 0.44606586716251873 | 0.43 |
| IID Gaussian prior | canonical_shellmean_high_energy_ratio | 3.1199122893463591 | 2.9472808122108289 | 3.321242409458756 | 3.12 |
| Full model | highband_error_relative_l2 | 0.87340655840460224 | 0.8621901011860641 | 0.88485340755612574 | 0.87 |
| IID Gaussian prior | highband_error_relative_l2 | 1.8252839107455296 | 1.7793718541521102 | 1.877885185559828 | 1.83 |

Source rows: `Dis_SI_Process/results/derived/20260910_1540/main_ablation_plot_coordinates.csv`. Exact input paths/hashes and schema filters are in source_manifest.json; full plotted artist coordinates and layout in main_artist_coordinates.json.

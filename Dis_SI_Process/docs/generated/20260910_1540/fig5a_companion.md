# Figure5a quantitative companion

Does the archived empirical conditional ensemble assign useful marginal predictive distributions?

Package B: archived benchmark checkpoint identities; a/b/d use200 paired states and64 draws with moving-block25/2,000 intervals; a/b bootstrap seed20260830, d method-specific recorded bootstrap_seed in benchmark_main_d.csv. f uses original1,000-state accuracy and separately measured resource protocols. These results were not remeasured for the new ablation reference.

Metric and aggregation: Normalized empirical CRPS = mean over points and four unobserved fields of [mean_k|X_k-y| - (1/(2K²))sum_kl|X_k-X_l|] / frozen training-field standard deviation. Ideal0; marginal, not phase-sensitive. Per-state values then equal-weight cohort mean. Boxes show state dispersion; mean intervals are separate.

Result and boundary: DMF-Gen has the lowest accepted normalized CRPS. This is not a demonstration of perfect calibration; inherited reliability remains underdispersed.

Coordinates at source precision:

| method | metric_value | ci_low | ci_high |
| --- | --- | --- | --- |
| DMF-Gen | 0.066717025886039102 | 0.063976462251938995 | 0.069406044216142104 |
| FFM-FNO | 0.39890982557542259 | 0.37387576073651241 | 0.43068150110374848 |
| FFM-Perceiver | 0.2596316273288849 | 0.2476113821739177 | 0.27228474277365861 |
| Latent FM | 0.3710934321594932 | 0.35437878333588901 | 0.38962117648171268 |
| SiT | 0.0999081668476242 | 0.0969710697923373 | 0.1030436406476502 |

Source rows: `Dis_SI_Process/results/derived/20260910_1540/benchmark_main_a_summary.csv`. Exact input paths/hashes and schema filters are in source_manifest.json; full plotted artist coordinates and layout in main_artist_coordinates.json.

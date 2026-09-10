# Figure5c quantitative companion

How do saved stochastic route/prior configurations compare in physical reconstruction?

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Metric and aggregation: Per-state physical field relative-L2 followed by equal-weight four-field macro and cohort mean; ideal0. Raw mean95%block20 intervals. Annotation100*(variant cohort mean/full cohort mean-1), not a mean of statewise ratios or percent-effect CI.

Result and boundary: Route-removal implementations have larger error. IID improves whole-field error. Local-only differs in optimized capacity and multiple routes, so this is not an isolated equal-capacity causal estimate.

Coordinates at source precision:

| display_label | policy | mean | block20_ci95_low | block20_ci95_high | annotation |
| --- | --- | --- | --- | --- | --- |
| Full model | last | 0.10632076677317751 | 0.10449524238782081 | 0.10822937860009681 | 0.1063 |
| No sensor feedback | last | 0.12697441583241301 | 0.12486830285874979 | 0.12924839657481199 | +19.4% |
| No local conditioning | last | 0.1447029438194137 | 0.14275715989634091 | 0.14666356018545801 | +36.1% |
| Local-only conditioning | last | 0.35495072396035698 | 0.35142188007896308 | 0.35869606554277261 | +233.8% |
| IID Gaussian prior | last | 0.1042938161915426 | 0.10241220413913921 | 0.10630154872937481 | -1.9% |

Source rows: `Dis_SI_Process/results/derived/20260910_1540/ablation_primary_summary.csv`. Exact input paths/hashes and schema filters are in source_manifest.json; full plotted artist coordinates and layout in main_artist_coordinates.json.

# Figure 5 V7R2 panel a: Normalized empirical CRPS

Scientific intent: Inherited marginal uncertainty evidence; preserve the accepted source coordinates, method order, points, box and interval treatment.

Quantitative definition: For K=64 accepted ensemble draws, CRPS is K^{-1} sum_k |X_k-y| - (2K^2)^{-1} sum_{k,l} |X_k-X_l|, normalized by the frozen training-field standard deviation. Points comprise the four unobserved fields and the equal macro-field mean over the accepted 200-state cohort.

Evidence package: Package B, inherited Figure 5 V6 release.

Visual design: Compact scatter and box summary; no decorative title; layout-only resizing and repositioning.

Inherited source coordinates and uncertainty treatment are preserved from the accepted Figure 5 V6 release.

Source ledger entries from `source_manifest.json`:

| key | path | sha256 | role |
| --- | --- | --- | --- |
| Dis_SI_Process/figures/generated/20260904_1200/fig5a_probabilistic_reconstruction_20260904_1200.svg | Dis_SI_Process/figures/generated/20260904_1200/fig5a_probabilistic_reconstruction_20260904_1200.svg | de4248f97c4041bbf468dbf1f1da64da476fc8d082baf8326d357137ca00bb55 | Package B inherited accepted 20260904_1200 source |


Output contract: `fig5a_v7r2_20260910_1707.svg` and `fig5a_v7r2_20260910_1707.png`.

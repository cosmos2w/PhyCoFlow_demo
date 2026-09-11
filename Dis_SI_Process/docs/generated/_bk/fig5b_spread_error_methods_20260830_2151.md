# Figure 5 V3 panel b: uq spread error methods

- Generated: `20260830_2151`
- SVG: `fig5b_spread_error_methods_20260830_2151.svg`
- Evidence status: **FORMAL**

## Scientific question

For each generative method, is macro normalized ensemble spread associated with macro ensemble-mean reconstruction error across states?

## Methods, cohort, and metric

Run `uq_compare_formal_20260830_v3r6`; 200 paired states; M=256; N=40,300; S=64; shared draw-ID seed schedule; moving-block bootstrap with block length 25 and 2000 replicates. The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights. Exact checkpoint identities are recorded in the run manifest.

## Main quantitative result

DMF-Gen ρ=0.654 [0.560, 0.721]; FFM-FNO ρ=0.183 [-0.004, 0.359]; FFM-Perceiver ρ=0.215 [0.080, 0.348]; Latent FM ρ=-0.033 [-0.164, 0.106]; SiT ρ=0.261 [0.103, 0.384].

## Exact source and run identity

`Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/spread_error_summary.csv`

Source classification: Five method-wise macro spread/error Spearman estimates.

## Caveats

Spearman association is descriptive and does not establish calibrated or prospective error prediction. The temporal moving-block bootstrap preserves local dependence in the single held-out trajectory.

## SI destination

Fieldwise spread/error scatter and bootstrap diagnostics.

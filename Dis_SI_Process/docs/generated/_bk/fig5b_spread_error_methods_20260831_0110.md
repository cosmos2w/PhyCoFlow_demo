# Figure 5 V4 panel b: uq spread error methods

- Generated: `20260831_0110`
- SVG: `fig5b_spread_error_methods_20260831_0110.svg`
- Evidence status: **FORMAL**

## Scientific question

For each generative method, is ensemble spread associated with reconstruction difficulty across the paired held-out states?

## Protocol and metric

Run `uq_compare_formal_20260830_v3r6`; 200 paired states; M=256; S=64; temporal moving-block bootstrap (25 blocks, 2000 replicates). The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights where applicable. Exact checkpoint identities are retained in source manifests.

## Main quantitative result

DMF-Gen ρ=0.654 [0.560, 0.721]; FFM-FNO ρ=0.183 [-0.004, 0.359]; FFM-Perceiver ρ=0.215 [0.080, 0.348]; Latent FM ρ=-0.033 [-0.164, 0.106]; SiT ρ=0.261 [0.103, 0.384].

## Exact source

`Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/spread_error_summary.csv`

Source classification: V3 formal macro spread/error association reused unchanged; not calibration.

## Interpretation limits

Spearman association is descriptive and does not establish calibration or prospective error prediction.

## SI destination

Fieldwise spread/error scatter and bootstrap diagnostics.

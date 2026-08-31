# Figure 5 V4 panel a: uq crps

- Generated: `20260831_0110`
- SVG: `fig5a_normalized_crps_20260831_0110.svg`
- Evidence status: **FORMAL**

## Scientific question

Which trained generative method produces the strongest empirical conditional ensemble under identical temperature-only measurements?

## Protocol and metric

Run `uq_compare_formal_20260830_v3r6`; 200 paired states; M=256; S=64; temporal moving-block bootstrap (25 blocks, 2000 replicates). The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights where applicable. Exact checkpoint identities are retained in source manifests.

## Main quantitative result

DMF-Gen 0.0667 [0.0640, 0.0694]; FFM-FNO 0.3989 [0.3739, 0.4307]; FFM-Perceiver 0.2596 [0.2476, 0.2723]; Latent FM 0.3711 [0.3544, 0.3896]; SiT 0.0999 [0.0970, 0.1030].

## Exact source

`Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/crps_summary.csv`

Source classification: V3 formal paired normalized empirical CRPS reused unchanged.

## Interpretation limits

Empirical finite-ensemble normalized CRPS; four unobserved fields are macro-averaged with equal 0.25 weight. Calibration and interval-width diagnostics remain SI-only.

## SI destination

Field-resolved CRPS, reliability, interval width and ensemble-diversity diagnostics.

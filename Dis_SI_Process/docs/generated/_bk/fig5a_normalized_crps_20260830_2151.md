# Figure 5 V3 panel a: uq crps

- Generated: `20260830_2151`
- SVG: `fig5a_normalized_crps_20260830_2151.svg`
- Evidence status: **FORMAL**

## Scientific question

Which trained generative method produces the strongest empirical conditional ensemble under identical temperature-only measurements?

## Methods, cohort, and metric

Run `uq_compare_formal_20260830_v3r6`; 200 paired states; M=256; N=40,300; S=64; shared draw-ID seed schedule; moving-block bootstrap with block length 25 and 2000 replicates. The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights. Exact checkpoint identities are recorded in the run manifest.

## Main quantitative result

DMF-Gen 0.0667 [0.0640, 0.0694]; FFM-FNO 0.3989 [0.3739, 0.4307]; FFM-Perceiver 0.2596 [0.2476, 0.2723]; Latent FM 0.3711 [0.3544, 0.3896]; SiT 0.0999 [0.0970, 0.1030].

## Exact source and run identity

`Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/crps_summary.csv`

Source classification: Five-method paired normalized empirical CRPS with equal field weights.

## Caveats

CRPS is empirical and finite-ensemble; four unobserved fields are normalized by frozen training standard deviations and receive equal 0.25 weight. Full reliability and interval-width curves remain SI-only.

## SI destination

Field-resolved CRPS, reliability curves, raw/normalized interval widths, ensemble diversity, and single-draw versus ensemble-mean error.

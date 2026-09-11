# Figure 5 V5 panel a

- SVG: `fig5a_probabilistic_reconstruction_v5_20260831_1409.svg`
- Scientific question: Is the complete conditional predictive distribution accurate?
- Reuse status: Reused unchanged from formal V3/V4.2; no inference or bootstrap rerun.

## Formal sources

- `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/per_state_method.csv`
- `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/crps_summary.csv`

## Cohort, checkpoints and statistics

Dataset/task: turbulent-combustion Cond_T missing-channel reconstruction; M=256; native N=40,300; unobserved fields Y_CH4, Y_CO, U1 and p are macro-averaged with equal 0.25 weight. The formal UQ cohort contains 200 paired temporal states and 64 shared-seed draws per state; 95% intervals use 2,000 moving-block-bootstrap replicates with block length 25. Exact adopted checkpoint SHA-256 identities are recorded in the source manifests.

## Metric

Pointwise empirical CRPS normalized by frozen training field standard deviation, averaged spatially, then macro-averaged equally across four unobserved fields.

## Main quantitative result

DMF-Gen: 0.0667; SiT: 0.0999; FFM-Perceiver: 0.2596; Latent FM: 0.3711; FFM-FNO: 0.3989

## Limitations and SI

CRPS assesses predictive-distribution quality but does not by itself establish calibration; formal reliability analysis shows underdispersion and is retained in SI. SI destination: `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/figures/generated/20260831_1409/si/fig5_si_calibration_interval_width_v5_20260831_1409.svg`.

## Storage / cleanup

No checkpoint, dataset, cache or old result bundle was copied. No raw bootstrap arrays or repeated inference stacks were retained in V5.

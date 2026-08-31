# Figure 5 V5 panel b

- SVG: `fig5b_uncertainty_difficult_states_v5_20260831_1409.svg`
- Scientific question: Does empirical ensemble spread distinguish easy and difficult held-out states?
- Reuse status: Reused unchanged from formal V3/V4.2; the exact bootstrap distribution is deterministically reconstructed from the adopted state table and seed.

## Formal sources

- `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/per_state_method.csv`
- `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/spread_error_summary.csv`

## Cohort, checkpoints and statistics

Dataset/task: turbulent-combustion Cond_T missing-channel reconstruction; M=256; native N=40,300; unobserved fields Y_CH4, Y_CO, U1 and p are macro-averaged with equal 0.25 weight. The formal UQ cohort contains 200 paired temporal states and 64 shared-seed draws per state; 95% intervals use 2,000 moving-block-bootstrap replicates with block length 25. Exact adopted checkpoint SHA-256 identities are recorded in the source manifests.

## Metric

Spearman association between macro normalized ensemble spread and macro ensemble-mean relative-L2 error.

## Main quantitative result

DMF-Gen: ρ=0.654; SiT: ρ=0.261; FFM-Perceiver: ρ=0.215; FFM-FNO: ρ=0.183; Latent FM: ρ=-0.033

## Limitations and SI

This is an association with reconstruction difficulty, not calibration, Bayesian posterior uncertainty, prospective error prediction, or causal evidence. SI destination: `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/figures/generated/20260831_1409/si/fig5_si_fieldwise_uq_v5_20260831_1409.svg`.

## Storage / cleanup

No checkpoint, dataset, cache or old result bundle was copied. No raw bootstrap arrays or repeated inference stacks were retained in V5.

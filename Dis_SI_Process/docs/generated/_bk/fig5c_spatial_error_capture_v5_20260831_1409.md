# Figure 5 V5 panel c

- SVG: `fig5c_spatial_error_capture_v5_20260831_1409.svg`
- Scientific question: Does empirical conditional ensemble uncertainty localize where reconstruction error occurs?
- Reuse status: New V5 streaming repeated-inference reducer; each state ensemble was reduced in memory and discarded, with no full stacks or per-draw files retained.

## Formal sources

- `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV5/UQLocalization/uq_localization_formal_v5/error_capture_curves.csv`
- `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV5/UQLocalization/uq_localization_formal_v5/error_capture_summary.csv`

## Cohort, checkpoints and statistics

Dataset/task: turbulent-combustion Cond_T missing-channel reconstruction; M=256; native N=40,300; unobserved fields Y_CH4, Y_CO, U1 and p are macro-averaged with equal 0.25 weight. The formal UQ cohort contains 200 paired temporal states and 64 shared-seed draws per state; 95% intervals use 2,000 moving-block-bootstrap replicates with block length 25. Exact adopted checkpoint SHA-256 identities are recorded in the source manifests.

## Metric

Within each state and field, locations are ranked by ensemble s.d.; captured absolute ensemble-mean error is evaluated at eight fractions, then fields are equally macro-averaged before temporal bootstrap.

## Main quantitative result

SiT: C(0.20)=0.631, EC-AUC=0.283; DMF-Gen: C(0.20)=0.570, EC-AUC=0.241; FFM-Perceiver: C(0.20)=0.555, EC-AUC=0.232; FFM-FNO: C(0.20)=0.522, EC-AUC=0.215; Latent FM: C(0.20)=0.510, EC-AUC=0.212

## Limitations and SI

The result validates spatial ranking informativeness on the held-out cohort, not prospective uncertainty calibration. SI destination: `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/figures/generated/20260831_1409/si/fig5_si_fieldwise_error_capture_v5_20260831_1409.svg`.

## Storage / cleanup

No checkpoint, dataset, cache or old result bundle was copied. No raw bootstrap arrays or repeated inference stacks were retained in V5.

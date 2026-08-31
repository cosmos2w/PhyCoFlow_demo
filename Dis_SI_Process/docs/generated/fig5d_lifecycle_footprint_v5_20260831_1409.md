# Figure 5 V5 panel d

- SVG: `fig5d_lifecycle_footprint_v5_20260831_1409.svg`
- Scientific question: What offline and online model-core footprint accompanies the frozen Figure-4 accuracy?
- Reuse status: New compact derivation from reused formal V3 native timings and V4/V4.2 canonical update replays; no benchmark or training replay was rerun.

## Formal sources

- `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV5/Lifecycle/lifecycle_formal_v5/lifecycle_summary.csv`
- `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV5/Lifecycle/lifecycle_formal_v5/lifecycle_stage_provenance.csv`

## Cohort, checkpoints and statistics

Dataset/task: turbulent-combustion Cond_T missing-channel reconstruction; M=256; native N=40,300; unobserved fields Y_CH4, Y_CO, U1 and p are macro-averaged with equal 0.25 weight. The formal UQ cohort contains 200 paired temporal states and 64 shared-seed draws per state; 95% intervals use 2,000 moving-block-bootstrap replicates with block length 25. Exact adopted checkpoint SHA-256 identities are recorded in the source manifests.

## Metric

x is accepted clean warm native latency; y is Replay-equivalent model-core training GPU-hours = sum(update ms × adopted updates × GPU count)/3.6e6; bubble area is frozen mean unobserved-field relative-L2.

## Main quantitative result

DMF-Gen: 16.69 ms, 62.5 GPU h, L2=0.117; FFM-FNO: 8.70 ms, 102.8 GPU h, L2=0.390; FFM-Perceiver: 23.09 ms, 55.9 GPU h, L2=0.348; Latent FM: 10.17 ms, 71.2 GPU h, L2=0.453; SiT: 20.99 ms, 516.0 GPU h, L2=0.210; MLP-RBF: 3.14 ms, 26.9 GPU h, L2=0.396; Geo-FNO: 3.41 ms, 119.7 GPU h, L2=0.230; Senseiver: 8.30 ms, 24.0 GPU h, L2=0.143

## Limitations and SI

The metric is not historical wall time or a matched-budget causal efficiency comparison; hardware, batch, solver and method-native configurations differ. SI destination: `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/figures/generated/20260831_1409/si/fig5_si_scalability_latency_memory_v5_20260831_1409.svg`.

## Storage / cleanup

No checkpoint, dataset, cache or old result bundle was copied. No raw bootstrap arrays or repeated inference stacks were retained in V5.

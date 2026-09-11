# Figure 5 V4 panel d: cost training compute

- Generated: `20260831_0110`
- SVG: `fig5d_accuracy_training_cost_20260831_0110.svg`
- Evidence status: **FORMAL**

## Scientific question

What computational investment is recorded for the adopted checkpoints, using the predeclared training-cost metric?

## Protocol and metric

Run `training_replay_formal_v4r2`; NVIDIA RTX 6000 Ada Generation; metric `training_update_time_ms`; batch policy `adopted canonical batch size for every stage`; 20 warmups and 100 measured updates per successful stage; all required stages were attempted. The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights where applicable. Exact checkpoint identities are retained in source manifests.

## Main quantitative result

DMF-Gen: error 0.1171, 528 ms/update; FFM-FNO: error 0.3898, 1.02e+03 ms/update; FFM-Perceiver: error 0.3479, 484 ms/update; SiT: error 0.2103, 659 ms/update; MLP-RBF: error 0.3962, 404 ms/update; Senseiver: error 0.1430, 479 ms/update. Method-level unavailable: Latent FM, Geo-FNO. Metric: training_update_time_ms.

## Exact source

`Dis_SI_Process/results/ValidationV4/TrainingCost/training_replay_formal_v4r2/training_cost_summary.csv`

Source classification: V4 training-compute source; no V3 query-latency fallback is permitted.

## Interpretation limits

Panel d is a descriptive canonical-configuration update-cost comparison, not total training GPU-hours or a causal matched-budget efficiency comparison. Adopted batch/query configurations differ by method; unavailable coordinates remain documented rather than imputed.

## SI destination

Training logs, stage-level versus total cost, replay validation, optimizer-update throughput and peak training memory.

# Figure 5 V4 panel c: cost native clean

- Generated: `20260831_0110`
- SVG: `fig5c_accuracy_latency_clean_20260831_0110.svg`
- Evidence status: **FORMAL**

## Scientific question

What native 40,300-point accuracy–cost trade-off is measured for the exact Figure 4 checkpoints?

## Protocol and metric

Run `formal_cost_clean_v3_20260830_v3`; NVIDIA RTX 6000 Ada Generation; float32, batch 1; timing boundary `warm_model_core_geometry_persisted`. The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights where applicable. Exact checkpoint identities are retained in source manifests.

## Main quantitative result

DMF-Gen: error 0.1171, 16.69 ms; FFM-FNO: error 0.3898, 8.70 ms; FFM-Perceiver: error 0.3479, 23.09 ms; Latent FM: error 0.4531, 10.17 ms; SiT: error 0.2103, 20.99 ms; MLP-RBF: error 0.3962, 3.14 ms; Geo-FNO: error 0.2299, 3.41 ms; Senseiver: error 0.1430, 8.30 ms.

## Exact source

`Dis_SI_Process/results/ValidationV3/CostClean/formal_cost_clean_v3_20260830_v3/native_summary.csv`

Source classification: V3 formal clean native model-core timing reused unchanged.

## Interpretation limits

Latency is clean-GPU, warm model-core timing and is hardware-, precision-, cache-, and boundary-specific. The plot does not imply unqualified Pareto superiority.

## SI destination

Full timing repeats, cold/no-cache timing, reserved memory, parameters, checkpoint sizes and timing-boundary audit.

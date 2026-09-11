# Figure 5 V3 panel c: cost native clean

- Generated: `20260830_2151`
- SVG: `fig5c_accuracy_latency_clean_20260830_2151.svg`
- Evidence status: **FORMAL**

## Scientific question

What native-mesh accuracy–latency trade-off is measured for the eight Figure 4 checkpoints under one clean model-core timing boundary?

## Methods, cohort, and metric

Run `formal_cost_clean_v3_20260830_v3`; NVIDIA RTX 6000 Ada Generation (GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39); driver 570.207; PyTorch 2.5.1+cu121; CUDA 12.1; batch 1 float32. Boundary `warm_model_core_geometry_persisted`; persistent cache: DMF reusable top-k sensor/query geometry only; value-dependent features recomputed per state. The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights. Exact checkpoint identities are recorded in the run manifest.

## Main quantitative result

DMF-Gen: error 0.1171, 16.69 ms; FFM-FNO: error 0.3898, 8.70 ms; FFM-Perceiver: error 0.3479, 23.09 ms; Latent FM: error 0.4531, 10.17 ms; SiT: error 0.2103, 20.99 ms; MLP-RBF: error 0.3962, 3.14 ms; Geo-FNO: error 0.2299, 3.41 ms; Senseiver: error 0.1430, 8.30 ms.

## Exact source and run identity

`Dis_SI_Process/results/ValidationV3/CostClean/formal_cost_clean_v3_20260830_v3/native_summary.csv`

Source classification: Eight exact checkpoints with clean-GPU model-core timing and frozen FieldL2.

## Caveats

Accuracy is reused from the exact-checkpoint 1,000-state FieldL2 table. Latency is hardware-, precision-, cache-, and timing-boundary-specific; no Pareto claim is forced.

## SI destination

Full repeats, p10/p90, cold-first timing, no-persistent-geometry timing, reserved memory, parameters, checkpoint sizes, and component/cache audit.

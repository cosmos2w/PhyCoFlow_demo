# Figure 5 V3 panel d: cost query scaling methods

- Generated: `20260830_2151`
- SVG: `fig5d_query_latency_20260830_2151.svg`
- Evidence status: **FORMAL**

## Scientific question

How does warm model-core latency scale with requested query count where the canonical model genuinely accepts variable query sets?

## Methods, cohort, and metric

Run `formal_cost_clean_v3_20260830_v3`; NVIDIA RTX 6000 Ada Generation (GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39); driver 570.207; PyTorch 2.5.1+cu121; CUDA 12.1; batch 1 float32. Boundary `warm_model_core_geometry_persisted`; persistent cache: DMF reusable top-k sensor/query geometry only; value-dependent features recomputed per state. The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights. Exact checkpoint identities are recorded in the run manifest.

## Main quantitative result

DMF-Gen 7.01→16.69; FFM-Perceiver 6.35→23.09; MLP-RBF 1.11→3.14; Senseiver 5.97→8.30 ms from 1,024 to 40,300 queries.

## Exact source and run identity

`Dis_SI_Process/results/ValidationV3/CostClean/formal_cost_clean_v3_20260830_v3/query_latency_summary.csv`

Source classification: Curves only for audited native variable-query models; fixed-grid methods are native-only.

## Caveats

A line denotes audited native variable-query execution. Open native-only markers denote fixed-discretization methods. No full-grid reconstruction followed by slicing is counted as query scaling.

## SI destination

All repeat timings and explicit failure/support reasons; no throughput-only points were generated.

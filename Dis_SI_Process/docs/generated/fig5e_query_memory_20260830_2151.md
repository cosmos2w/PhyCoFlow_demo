# Figure 5 V3 panel e: cost memory scaling methods

- Generated: `20260830_2151`
- SVG: `fig5e_query_memory_20260830_2151.svg`
- Evidence status: **FORMAL**

## Scientific question

How does peak allocated GPU memory scale under the identical query-support protocol used for panel d?

## Methods, cohort, and metric

Run `formal_cost_clean_v3_20260830_v3`; NVIDIA RTX 6000 Ada Generation (GPU-f6a4ddbb-ad44-5ef5-0421-eecf7120df39); driver 570.207; PyTorch 2.5.1+cu121; CUDA 12.1; batch 1 float32. Boundary `warm_model_core_geometry_persisted`; persistent cache: DMF reusable top-k sensor/query geometry only; value-dependent features recomputed per state. The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights. Exact checkpoint identities are recorded in the run manifest.

## Main quantitative result

DMF-Gen 70.68→418.41; FFM-Perceiver 44.88→315.31; MLP-RBF 46.10→1087.53; Senseiver 195.07→1182.59 MiB from 1,024 to 40,300 queries.

## Exact source and run identity

`Dis_SI_Process/results/ValidationV3/CostClean/formal_cost_clean_v3_20260830_v3/memory_summary.csv`

Source classification: Peak allocated memory under the identical support/query protocol as panel d.

## Caveats

Peak allocated—not reserved—memory is shown. Model weights, one prepared device-side state, and the allowed reusable geometry cache are included; the throughput-only extension was not run.

## SI destination

Peak reserved memory and allocation details under the same N/support matrix.

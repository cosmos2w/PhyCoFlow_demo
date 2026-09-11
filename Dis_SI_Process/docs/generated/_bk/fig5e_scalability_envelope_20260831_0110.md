# Figure 5 V4 panel e: cost scalability envelope

- Generated: `20260831_0110`
- SVG: `fig5e_scalability_envelope_20260831_0110.svg`
- Evidence status: **FORMAL**

## Scientific question

How do warm latency and allocated memory scale when the requested query set extends beyond the native grid?

## Protocol and metric

Run `scale_stress_formal_v4`; sensor-prefixed Sobol query specification hash `ac4bed171d6536667a2d1898819ec0e3350e5e751d5cf74db72ebce95cfd6a9a`; native and throughput-only regions are explicitly separated. V3 native prefix: `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV3/CostClean/formal_cost_clean_v3_20260830_v3/query_latency_summary.csv` + `/home/wanglz/Desktop/src/PhyCoFlow/Dis_SI_Process/results/ValidationV3/CostClean/formal_cost_clean_v3_20260830_v3/memory_summary.csv`. The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights where applicable. Exact checkpoint identities are retained in source manifests.

## Main quantitative result

Variable-query curves: DMF-Gen, FFM-Perceiver, MLP-RBF, Senseiver. Largest successful N: DMF-Gen=8,000,000; FFM-Perceiver=4,000,000; MLP-RBF=1,000,000; Senseiver=1,000,000.

## Exact source

`Dis_SI_Process/results/ValidationV4/ScaleStress/scale_stress_formal_v4/scale_stress_summary.csv`

Source classification: V4 Sobol high-N latency/memory stress source merged with validated V3 native rows; common query hashes and explicit throughput-only region are enforced.

## Interpretation limits

Only canonical variable-query paths receive curves. Values above N=40,300 are throughput-only stress measurements and carry no physical accuracy claim.

## SI destination

Coordinate specification/hash, geometry preparation, reserved memory, OOM/runtime-cap table and first-failure details.

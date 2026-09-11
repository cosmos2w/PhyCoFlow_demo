# Figure 5 panel e: cost query memory

- Generated: `20260830_1907`
- SVG: `fig5e_query_memory_20260830_1907.svg`
- Evidence status: **FORMAL**

## Scientific question

Measure DMF-Gen latency and peak allocated memory over real-coordinate query sizes.

## Main quantitative result

From N=1,024 to 40,300, median latency changes 5.84× and peak allocated memory changes 2.71×.

## Source and identity

`0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Process_Results/ValidationV2/Cost/formal_cost_20260830_v2/benchmark_summary.csv`

Run `formal_cost_20260830_v2`; schema `validation-v2-cost-1`; plan SHA-256 `06af0715d3e45576cd8406741c28fb41b8c2e12b440d388ccd020c5d53f746c2`; formal flag `True`.

## Uncertainty definition and caveats

Each point uses real-coordinate inference with the same M=256 conditioning sensors, synchronized CUDA timing for at least 10 s, and peak allocated—not reserved—memory. Scaling is hardware- and chunk-size-specific. Source classification: Formal DMF real-coordinate query/memory sweep.

## SI destination

All repeat timings, chunking settings, allocated and reserved memory, cache-equivalence diagnostics, and device metadata.

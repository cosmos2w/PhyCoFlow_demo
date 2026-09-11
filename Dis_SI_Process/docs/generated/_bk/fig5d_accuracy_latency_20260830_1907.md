# Figure 5 panel d: cost native pareto

- Generated: `20260830_1907`
- SVG: `fig5d_accuracy_latency_20260830_1907.svg`
- Evidence status: **FORMAL**

## Scientific question

Compare native-mesh accuracy and synchronized warm latency for the eight Figure 4 methods.

## Main quantitative result

DMF-Gen: 0.1171 at 127.05 ms; FFM-FNO: 0.3898 at 20.42 ms; FFM-Perceiver: 0.3479 at 35.07 ms; Latent FM: 0.4531 at 24.50 ms; SiT: 0.2103 at 41.18 ms; MLP-RBF: 0.3962 at 13.22 ms; Geo-FNO: 0.2299 at 13.73 ms; Senseiver: 0.1430 at 17.67 ms.

## Source and identity

`0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Process_Results/ValidationV2/Cost/formal_cost_20260830_v2/benchmark_summary.csv`

Run `formal_cost_20260830_v2`; schema `validation-v2-cost-1`; plan SHA-256 `06af0715d3e45576cd8406741c28fb41b8c2e12b440d388ccd020c5d53f746c2`; formal flag `True`.

## Uncertainty definition and caveats

Accuracy is the frozen 1,000-state FieldL2 estimate with state-bootstrap intervals; latency is synchronized warm inference IQR after 10 warm-ups and at least 10 s of timing. Absolute latency is hardware- and adapter-specific. Source classification: Formal canonical-adapter native benchmark with frozen FieldL2 join.

## SI destination

Per-method checkpoint hashes, adapters, repeat timings, error bootstrap tables, warm-up policy, and unavailable/failure handling.

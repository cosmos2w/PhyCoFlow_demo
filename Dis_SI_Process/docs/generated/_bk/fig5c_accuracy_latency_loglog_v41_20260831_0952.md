# Figure 5 V4.1 panel c

- Generated: `20260831_0952`
- SVG: `fig5c_accuracy_latency_loglog_v41_20260831_0952.svg`
- Evidence status: **FORMAL**

## Protocol and visual statistic

Exact V3 clean warm model-core timing at N=40,300 and frozen 1,000-state field error; both axes are logarithmic.

## Main quantitative result

DMF-Gen: error=0.1171, latency=16.69 ms; FFM-FNO: error=0.3898, latency=8.70 ms; FFM-Perceiver: error=0.3479, latency=23.09 ms; Latent FM: error=0.4531, latency=10.17 ms; SiT: error=0.2103, latency=20.99 ms; MLP-RBF: error=0.3962, latency=3.14 ms; Geo-FNO: error=0.2299, latency=3.41 ms; Senseiver: error=0.1430, latency=8.30 ms.

## Exact source

`Dis_SI_Process/results/ValidationV3/CostClean/formal_cost_clean_v3_20260830_v3/native_summary.csv`

## Interpretation limit

Panel a shows the distribution across paired states, whereas panel b shows uncertainty in a method-level association statistic. Panel d compares adopted configurations, not a causal matched-budget training experiment. Panel e carries no accuracy claim beyond 40,300 points.

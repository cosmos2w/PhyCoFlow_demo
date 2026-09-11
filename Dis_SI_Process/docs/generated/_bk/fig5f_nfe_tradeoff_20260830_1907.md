# Figure 5 panel f: cost nfe tradeoff

- Generated: `20260830_1907`
- SVG: `fig5f_nfe_tradeoff_20260830_1907.svg`
- Evidence status: **FORMAL**

## Scientific question

Trace DMF-Gen accuracy and synchronized latency as measured vector-field evaluations increase.

## Main quantitative result

NFE 1: error 0.1133, latency 106.29 ms; NFE 2: error 0.1175, latency 114.01 ms; NFE 4: error 0.1267, latency 140.48 ms; NFE 8: error 0.1349, latency 190.51 ms.

## Source and identity

`0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Process_Results/ValidationV2/Cost/formal_cost_20260830_v2/nfe_error.csv`

Run `formal_cost_20260830_v2`; schema `validation-v2-cost-1`; plan SHA-256 `06af0715d3e45576cd8406741c28fb41b8c2e12b440d388ccd020c5d53f746c2`; formal flag `True`.

## Uncertainty definition and caveats

Errors use the same predeclared 50-state cohort and common generation seeds at every measured NFE; error bars are state-bootstrap intervals and latency bars are repeat IQRs. The observed worsening with NFE is reported without assuming monotonic solver improvement. Source classification: Formal DMF measured-NFE accuracy/latency sweep.

## SI destination

Per-state errors, common seeds, vector-field-call accounting, repeat timings, solver settings, and error-bootstrap tables.

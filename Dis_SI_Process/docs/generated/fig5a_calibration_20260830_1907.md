# Figure 5 panel a: uq calibration

- Generated: `20260830_1907`
- SVG: `fig5a_calibration_20260830_1907.svg`
- Evidence status: **FORMAL**

## Scientific question

Measure whether repeated generations attain nominal central-interval coverage.

## Main quantitative result

Y_CH4: mean absolute calibration error 0.219; Y_CO: mean absolute calibration error 0.390; U1: mean absolute calibration error 0.579; p: mean absolute calibration error 0.659.

## Source and identity

`0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Process_Results/ValidationV2/Uncertainty/u2_formal_20260830_v1/coverage_by_level.csv`

Run `u2_formal_20260830_v1`; schema `validation-v2-uncertainty-1`; plan SHA-256 `06af0715d3e45576cd8406741c28fb41b8c2e12b440d388ccd020c5d53f746c2`; formal flag `True`.

## Uncertainty definition and caveats

Intervals are empirical central intervals from 64 draws on the frozen 200-state U2 cohort; moving-block bootstrap intervals preserve local temporal dependence. The severe undercoverage means the raw ensemble must not be described as calibrated. Source classification: Formal U2 state-level empirical coverage.

## SI destination

Field-unit coverage counts, per-state interval membership, bootstrap settings, and the U3 sensor-density calibration sweep.

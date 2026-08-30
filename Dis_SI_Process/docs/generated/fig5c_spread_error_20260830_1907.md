# Figure 5 panel c: uq spread error

- Generated: `20260830_1907`
- SVG: `fig5c_spread_error_20260830_1907.svg`
- Evidence status: **FORMAL**

## Scientific question

Test whether larger state-level ensemble spread is associated with larger reconstruction error.

## Main quantitative result

Spearman associations are Y_CH4 0.631, Y_CO 0.567, U1 0.221, p 0.659.

## Source and identity

`0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Process_Results/ValidationV2/Uncertainty/u1_formal_20260830_v1/per_state_field.csv`

Run `u1_formal_20260830_v1`; schema `validation-v2-uncertainty-1`; plan SHA-256 `06af0715d3e45576cd8406741c28fb41b8c2e12b440d388ccd020c5d53f746c2`; formal flag `True`.

## Uncertainty definition and caveats

Spearman statistics and confidence intervals use the frozen 1,000-state U1 cohort and temporal moving-block bootstrap. Association is descriptive, field dependent, and does not establish prospective error prediction or calibration. Source classification: Formal U1 state-level spread/error association.

## SI destination

Full state scatter, binned counts, Pearson correlations, ensemble-diversity diagnostics, and predeclared visual cases.

# Figure 5 panel a: uq map

- Generated: `20260830_1644`
- SVG: `fig5_panel_a_uq_map_20260830_1644.svg`
- Evidence status: **PROXY**

## Purpose and meaning

Localize reconstruction fidelity and empirical spread for one unobserved field.

## Main quantitative result

The displayed reconstruction has relative L2 = 0.4110; the 99th percentile of the displayed spread/sensitivity field is 0.005275.

## Source data / generation source

`0_demo_TurbulentCombustion/_CheckNotes/Stage6_formal_baseline/evaluation/matched_reconstruction/F0_best/nfe2.npz`

The build reads this source in place and writes only a lightweight derived summary under `Dis_SI_Process/results/`; no raw result or checkpoint is copied.

## Caveats and draft status

Deterministic NFE=2 reconstruction; cross-NFE standard deviation is a solver-sensitivity proxy. This panel is a layout/engineering proxy and must not be cited as Figure 5 manuscript evidence.

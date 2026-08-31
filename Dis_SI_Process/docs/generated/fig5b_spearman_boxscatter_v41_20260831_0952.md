# Figure 5 V4.1 panel b

- Generated: `20260831_0952`
- SVG: `fig5b_spearman_boxscatter_v41_20260831_0952.svg`
- Evidence status: **FORMAL**

## Protocol and visual statistic

Box/scatter uses 2,000 predeclared temporal moving-block-bootstrap Spearman replicates per method (block length 25). The open marker is the full-sample ρ; bootstrap replicates are not independent test states.

## Main quantitative result

DMF-Gen ρ=0.654 [0.560, 0.721]; FFM-FNO ρ=0.183 [-0.004, 0.359]; FFM-Perceiver ρ=0.215 [0.080, 0.348]; Latent FM ρ=-0.033 [-0.164, 0.106]; SiT ρ=0.261 [0.103, 0.384].

## Exact source

`Dis_SI_Process/results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6/per_state_method.csv`

## Interpretation limit

Panel a shows the distribution across paired states, whereas panel b shows uncertainty in a method-level association statistic. Panel d compares adopted configurations, not a causal matched-budget training experiment. Panel e carries no accuracy claim beyond 40,300 points.

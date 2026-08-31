# Zero-H-matched Figure 5 V4.2 panel b

- SVG: `fig5b_zeroh_spread_error_matched_v42_20260831_1056.svg`
- Evidence: **strict formal**
- Model coverage: DMF-Gen and FFM-Perceiver, the two stochastic models adopted in this scenario.

## Protocol

Method-wise Spearman association between normalized spatial RMS ensemble spread and physical ensemble-mean relative L2. The box/scatter shows 2,000 unique-case bootstrap estimates; the open marker is the full-sample rho.

## Main result

DMF-Gen: rho=0.359 [0.207, 0.490]; FFM-Perceiver: rho=0.784 [0.719, 0.834].

## Exact source

`Dis_SI_Process/results/ValidationV42/ZeroHMatched/UQ/zeroh_uq_formal_v42/spread_error_summary.csv`

This is the single-density `4_ZeroH_Balanced` scenario, not Cond_T. Deterministic models are excluded from panels a/b because panel b requires nonzero ensemble spread; no missing metric is imputed. Panel c uses the archive's legacy full sampling path and synchronized wall timing, without the persistent DMF top-k geometry cache used by the Cond_T portable core; its absolute latency is not a cross-scenario comparison.

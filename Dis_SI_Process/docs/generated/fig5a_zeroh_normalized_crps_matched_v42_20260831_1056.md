# Zero-H-matched Figure 5 V4.2 panel a

- SVG: `fig5a_zeroh_normalized_crps_matched_v42_20260831_1056.svg`
- Evidence: **strict formal**
- Model coverage: DMF-Gen and FFM-Perceiver, the two stochastic models adopted in this scenario.

## Protocol

State-level normalized empirical CRPS for 200 paired unique-case/time states and 64 draws/state. The box/scatter shows states; the open marker and line show the formal mean and 2,000-replicate 95% case-bootstrap CI.

## Main result

DMF-Gen: mean CRPS=0.0171 [0.0148, 0.0197]; FFM-Perceiver: mean CRPS=0.0460 [0.0402, 0.0524].

## Exact source

`Dis_SI_Process/results/ValidationV42/ZeroHMatched/UQ/zeroh_uq_formal_v42/per_state_method.csv`

This is the single-density `4_ZeroH_Balanced` scenario, not Cond_T. Deterministic models are excluded from panels a/b because panel b requires nonzero ensemble spread; no missing metric is imputed. Panel c uses the archive's legacy full sampling path and synchronized wall timing, without the persistent DMF top-k geometry cache used by the Cond_T portable core; its absolute latency is not a cross-scenario comparison.

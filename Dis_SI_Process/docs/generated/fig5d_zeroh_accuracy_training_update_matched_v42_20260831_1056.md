# Zero-H-matched Figure 5 V4.2 panel d

- SVG: `fig5d_zeroh_accuracy_training_update_matched_v42_20260831_1056.svg`
- Evidence: **strict formal**
- Model coverage: DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver, the four adopted best checkpoints.

## Protocol

The same audited accuracy versus canonical L/M training-update time at batch 512. The plotted cost is the equal-weight mean of L- and M-resolution median synchronized wall time. Both axes are logarithmic.

## Main result

DMF-Gen: error=0.0405, training update=331.50 ms; FFM-Perceiver: error=0.1023, training update=328.40 ms; MLP-RBF: error=0.1808, training update=169.85 ms; Senseiver: error=0.0862, training update=320.31 ms.

## Exact source

`Dis_SI_Process/results/ValidationV42/ZeroHMatched/Cost/zeroh_cost_formal_v42/training_update_summary.csv`

This is the single-density `4_ZeroH_Balanced` scenario, not Cond_T. Deterministic models are excluded from panels a/b because panel b requires nonzero ensemble spread; no missing metric is imputed. Panel c uses the archive's legacy full sampling path and synchronized wall timing, without the persistent DMF top-k geometry cache used by the Cond_T portable core; its absolute latency is not a cross-scenario comparison.

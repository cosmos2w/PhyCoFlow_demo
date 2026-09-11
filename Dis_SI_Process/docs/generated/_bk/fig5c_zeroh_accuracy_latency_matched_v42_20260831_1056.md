# Zero-H-matched Figure 5 V4.2 panel c

- SVG: `fig5c_zeroh_accuracy_latency_matched_v42_20260831_1056.svg`
- Evidence: **strict formal**
- Model coverage: DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver, the four adopted best checkpoints.

## Protocol

Audited 300-case mean density relative L2 versus clean warm model-core native inference latency at N=16,384 and 256 sensors. Both axes are logarithmic.

## Main result

DMF-Gen: error=0.0405, latency=40.82 ms; FFM-Perceiver: error=0.1023, latency=18.73 ms; MLP-RBF: error=0.1808, latency=1.78 ms; Senseiver: error=0.0862, latency=10.16 ms.

## Exact source

`Dis_SI_Process/results/ValidationV42/ZeroHMatched/Cost/zeroh_cost_formal_v42/native_cost_summary.csv`

This is the single-density `4_ZeroH_Balanced` scenario, not Cond_T. Deterministic models are excluded from panels a/b because panel b requires nonzero ensemble spread; no missing metric is imputed. Panel c uses the archive's legacy full sampling path and synchronized wall timing, without the persistent DMF top-k geometry cache used by the Cond_T portable core; its absolute latency is not a cross-scenario comparison.

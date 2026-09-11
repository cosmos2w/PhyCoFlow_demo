# Metric-matched Zero-H Figure 5 V4.2 completion report

- Generated: `20260831_1056`
- Git commit at build: `6d4f26a83604db42b29de374fda26a49ab385cbf`
- Formal UQ run: `zeroh_uq_formal_v42` on `cuda:2`
- Formal cost run: `zeroh_cost_formal_v42` on `cuda:1`
- Export QA: **PASS**

## Why V4.1 differed

The V4.1 backup was intentionally limited to the four reconstruction-accuracy distributions already present in the archive. At that point, the Zero-H scenario had no cross-model ensemble UQ or clean cost evidence, so matching the main Figure 5 semantics would have required imputing unavailable metrics. V4.2 adds the minimum formal scenario-specific measurements and therefore can use the same panel meanings.

## Model coverage

- Panels a/b: DMF-Gen and FFM-Perceiver. These are the only adopted stochastic models in `4_ZeroH_Balanced`.
- Panels c/d: DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver.
- MLP-RBF and Senseiver are excluded from panel b because deterministic zero spread makes Spearman undefined, not because their results were discarded.

## Results

- Panel a: DMF-Gen: mean CRPS=0.0171 [0.0148, 0.0197]; FFM-Perceiver: mean CRPS=0.0460 [0.0402, 0.0524].
- Panel b: DMF-Gen: rho=0.359 [0.207, 0.490]; FFM-Perceiver: rho=0.784 [0.719, 0.834].
- Panel c: DMF-Gen: error=0.0405, latency=40.82 ms; FFM-Perceiver: error=0.1023, latency=18.73 ms; MLP-RBF: error=0.1808, latency=1.78 ms; Senseiver: error=0.0862, latency=10.16 ms.
- Panel d: DMF-Gen: error=0.0405, training update=331.50 ms; FFM-Perceiver: error=0.1023, training update=328.40 ms; MLP-RBF: error=0.1808, training update=169.85 ms; Senseiver: error=0.0862, training update=320.31 ms.

The task is density reconstruction at native `N=16,384` with 256 observations. These values must not be compared numerically as though they were Cond_T four-field metrics at `N=40,300`.

## Latency-boundary caveat

The Zero-H inference runner uses the super-resolution archive's legacy full `sample()` implementation and synchronized wall timing. It does not expose the persistent top-k geometry/static-feature cache used by the Cond_T portable core. The panel-c coordinate is valid for the four Zero-H checkpoints under this shared runner, but the 40.82-ms DMF-Gen value is not an optimized cross-scenario counterpart to the cached Cond_T value.

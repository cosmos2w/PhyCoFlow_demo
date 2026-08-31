# Figure 5 validation V3 contract

Core conclusion: conditional ensemble quality differs across the five trained generative reconstruction methods, while clean native accuracy–latency and support-qualified query scaling expose distinct deployment trade-offs without asserting Pareto superiority.

- Schema: `figure5-validation-v3`.
- Archetype: compact quantitative grid with an asymmetric bottom-row hero panel.
- Target/output: Nature Machine Intelligence-style, 183 mm × 118 mm, editable SVG only.
- Backend: Python/Matplotlib exclusively for plotting, previewing, export, and visual QA.
- Top row: paired generative-method comparison (`a` normalized empirical CRPS; `b` method-wise macro spread/error Spearman association).
- Bottom row: computational characteristics (`c` corrected native accuracy–latency hero panel; `d` query-count latency; `e` peak allocated memory).
- Formal unobserved fields: `Y_CH4`, `Y_CO`, `U1`, and `p`, each with macro weight 0.25 under `Cond_T`.
- Generative order: DMF-Gen, FFM-FNO, FFM-Perceiver, Latent FM, SiT.
- Eight-method order: DMF-Gen, FFM-FNO, FFM-Perceiver, Latent FM, SiT, MLP-RBF, Geo-FNO, Senseiver.
- Source-data policy: reuse the matching DMF U2 summaries and frozen Figure 4 FieldL2 products in place. Do not copy checkpoints, datasets, reconstruction caches, or V2 result bundles.
- Strict behavior: `--strict-formal` fails on missing cross-model UQ, missing/failed V3 QA, checkpoint mismatch, V2 cost input, unsupported query-scaling rows, or unresolved DMF timing reconciliation. It never substitutes V2 timing or proxy data.

## Panel evidence map

### a — Normalized empirical CRPS

Use the fixed 200-state cohort, exact saved M=256 temperature sensor plan, S=64 shared draw-ID seeds, and each adopted Figure 4 generative checkpoint. Compute empirical CRPS pointwise after normalization by the frozen training field standard deviation, average spatially, then equal-weight the four unobserved fields. Report the state mean and temporal moving-block-bootstrap 95% CI. Lower is better.

### b — Macro spread/error association

For every state and generative method, equal-weight normalized spatial RMS ensemble standard deviation across the four unobserved fields and separately equal-weight ensemble-mean physical relative-L2. Report Spearman association across the paired cohort with temporal moving-block-bootstrap 95% CI. Use “associated with reconstruction difficulty,” never “predicts error.”

### c — Corrected native accuracy–latency

At M=256 and N=40,300, plot clean-GPU warm model-core median latency (log x; repeat IQR) against frozen 1,000-state mean unobserved-field relative-L2 (temporal-bootstrap 95% CI) for all eight exact checkpoints. The timer excludes loading, data I/O, CPU preprocessing, host transfer, generic adapter dispatch, metrics, device-to-host transfer, plotting, and disk I/O. It includes stochastic noise, value-dependent conditioning, every model/flow evaluation, adopted observation consistency, and device-side output. Only reusable state-independent sensor/query geometry may persist. DMF uses the canonical configured 8,192-point reconstruction chunk. Its unified timer must agree within 20% with the same-setting direct and independent exact-shape timers; the approximately 29 ms prior probe must be mapped to a documented profiled boundary before promotion.

### d — Query-count latency

Use N=1,024, 4,096, 16,384, and 40,300 only for canonical methods that natively accept variable query coordinates. Fixed-discretization models receive an open native-size marker at N=40,300 only. Never reconstruct the full grid and slice it to claim scaling. Latency uses the same clean model-core definition as panel c.

### e — Query-count peak allocated memory

Use exactly the same method eligibility and N keys as panel d. Reset CUDA peak-memory statistics immediately before one measured core inference and report peak allocated memory. Peak reserved memory remains SI-only.

## Evidence hierarchy and review risks

- Hero evidence: panel c, because it corrects the invalid V2 timing coordinate while retaining exact Figure 4 accuracy.
- Validation evidence: panels a and b establish paired cross-model ensemble quality.
- Deployment evidence: panels d and e distinguish genuine query evaluation from fixed-grid inference.
- SI/robustness: full calibration/width curves, fieldwise UQ, diversity, cold/no-cache timing, reserved memory, component timing, and NFE/solver diagnostics.
- Reviewer risks: finite-ensemble CRPS, temporal dependence, stochastic seed control, method-native solver/NFE differences, exact checkpoint joins, cache fairness, and accidental full-grid slicing.
- Image integrity: all panels are vector quantitative plots; no raster image adjustment or representative-case selection enters the main figure.

No ablation training and no optional 100k–1M throughput-only extension are part of this contract.

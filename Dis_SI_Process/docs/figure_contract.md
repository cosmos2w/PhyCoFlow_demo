# Figure 5 validation V4 contract

Core conclusion: DMF-Gen provides the strongest measured conditional-distribution quality, while separate online-inference, offline-training, and support-qualified high-resolution benchmarks expose transparent lifecycle trade-offs without asserting universal efficiency or accuracy beyond the native domain.

- Schema: `figure5-validation-v4`.
- Archetype: quantitative grid with a full-width scalability hero panel.
- Target/output: Nature Machine Intelligence-style, 183 mm × 138 mm, Python/Matplotlib only, editable SVG primary output with Python-rendered print-size preview for QA.
- Formal unobserved fields: `Y_CH4`, `Y_CO`, `U1`, and `p`, each with macro weight 0.25 under `Cond_T`.
- Generative order: DMF-Gen, FFM-FNO, FFM-Perceiver, Latent FM, SiT.
- Eight-method order: DMF-Gen, FFM-FNO, FFM-Perceiver, Latent FM, SiT, MLP-RBF, Geo-FNO, Senseiver.
- Visual identity: exact Figure 3/4 method colors and one stable method marker across panels c--e.
- Provenance: reuse the unchanged formal V3 UQ and clean native-cost products in place; write all new training-cost, scale-stress, figure, manifest, and QA products under new ValidationV4 roots.
- Strict behavior: `--strict-formal` fails on unsupported training-cost estimates, missing stage/update provenance, fake query scaling, non-common stress coordinates, V2 timing, failed V3 identities, missing V4 QA, or proxy data.

## Evidence chain and panel map

### a — Probabilistic reconstruction quality

Reuse formal V3 normalized empirical CRPS from the fixed 200-state, M=256, S=64 paired cohort. Compute CRPS pointwise after normalization by frozen training field standard deviation, average spatially, then equal-weight the four unobserved fields. Show the state mean and temporal moving-block-bootstrap 95% CI in a horizontal forest plot. Lower is better.

### b — Uncertainty informativeness

Reuse formal V3 method-wise Spearman association between macro normalized ensemble spread and macro ensemble-mean unobserved-field error, with temporal moving-block-bootstrap 95% CI. Describe the result only as uncertainty informativeness or association with reconstruction difficulty; it is not evidence of calibration or prospective error prediction.

Panels a and b share method rows, row guides, marker/color grammar, and a faint DMF-Gen row highlight. Method labels appear only in panel a.

### c — Native inference accuracy--cost

Reuse the validated V3 clean-GPU native benchmark at M=256, N=40,300, batch 1, float32 for all eight exact Figure 4 checkpoints. Plot frozen 1,000-state mean unobserved-field relative-L2 against warm model-core median latency with temporal-bootstrap accuracy CI and repeat IQR. The timer excludes loading, data I/O, CPU preparation, host transfer, generic adapter dispatch, metrics, device-to-host transfer, plotting, and disk I/O. It includes stochastic initialization, value-dependent conditioning, every required model/flow evaluation, observation consistency, and device-side output. Only reusable value-independent sensor/query geometry may persist.

### d — Offline training accuracy--cost

Use the same frozen reconstruction-error coordinate and y limits as panel c. The x coordinate must be frozen before ranking methods and must be one of:

1. explicit total GPU-hours to the adopted checkpoint with known hardware, active GPU count, optimizer updates, and every required training stage; or
2. replay-equivalent GPU-hours derived from a standardized clean-GPU forward/backward/optimizer replay and the adopted checkpoint's documented update count, promoted only after agreement with trustworthy historical records within the predeclared tolerance; or
3. a directly measured per-update training-compute metric if total/update provenance is insufficient.

Filesystem modification times are never evidence of training duration. Every required stage of a multi-stage method must be measured and reported; if unlike stages cannot be reduced to the selected direct metric without an arbitrary aggregation, the method-level coordinate remains unavailable and its stage values stay in SI. The panel is a descriptive footprint of adopted checkpoints, not a matched-budget causal efficiency ablation.

### e — High-resolution scalability envelope

Use two vertically aligned axes sharing query count: warm model-core latency and peak allocated GPU memory. Native-validated real-coordinate points are N=1,024, 4,096, 16,384, and 40,300. Only canonically arbitrary-query methods receive curves. Fixed-discretization methods receive open native-only markers at N=40,300; full-grid reconstruction followed by slicing is forbidden.

For N>40,300, use one frozen, hashed, deterministic coordinate specification shared by all eligible methods. Predeclare N=100k, 250k, 500k, 1M, 2M, and 4M, with method-independent adaptive continuation rules. This region is shaded and labelled `throughput-only stress test`; it carries no physical-accuracy or super-resolution claim. Stop at the first CUDA OOM, 90% physical-VRAM allocation boundary, runtime cap, or global safety cap, and record both largest successful N and first failed N. First-use geometry preparation and peak reserved memory remain SI-only.

## Evidence hierarchy

- Primary scientific evidence: panel a, conditional-distribution quality across five trained generators.
- Uncertainty validation: panel b, state-level informativeness without a calibration claim.
- Lifecycle trade-off evidence: paired panels c and d, with shared error geometry and one consistent legend.
- Deployment hero evidence: panel e, which distinguishes canonical query evaluation from fixed-grid inference and separates the native accuracy domain from throughput-only stress.
- SI/robustness: full reliability/width curves, fieldwise UQ, diversity, cold/no-cache timing, training provenance details, replay throughput/memory, first-use geometry cost, peak reserved memory, full failure table, and NFE/solver diagnostics.

## Statistics and source-data requirements

- UQ unit: one held-out temporal state; 200 paired states, 64 shared draw-ID seeds, moving-block bootstrap with the frozen V3 block length and replicate count.
- Accuracy unit: one held-out temporal state from the frozen 1,000-state Figure 4 cohort; temporal moving-block-bootstrap 95% CI.
- Latency: synchronized clean-GPU warm repeats, median and IQR, with hardware/software and timing boundary recorded.
- Memory: peak allocated CUDA memory reset immediately before one measured core inference; reserved memory reported only in SI.
- Training cost: exact stage, device, adopted canonical batch/query configuration, update count, timing statistic, replay duration, and historical-validation status recorded per method. The promoted direct fallback is synchronized update time at each adopted configuration, with 20 warmups, 100 measured updates per successful stage, and a 25% early/late block-stability gate. Cross-method values are descriptive footprints, not batch-normalized or matched-budget causal estimates; failed canonical replays and unlike multi-stage methods remain explicitly unavailable.
- Every quantitative panel must be reproducible from a timestamp-matched source table and manifest.

## Visual and image-integrity contract

- Explicit GridSpec geometry; no large row headers.
- Panels a/b and c/d use shared-axis logic without duplicate adjacent y labels.
- Panel e spans the full width with aligned latency/memory axes, a native boundary, and restrained throughput shading.
- Panel letters are lowercase bold; text remains editable SVG text; all labels and legends must pass inspection at 183-mm print width.
- All panels are vector quantitative plots. No raster image adjustment or representative-case selection enters the figure.

## Reviewer risks and non-claims

- Finite-ensemble CRPS and temporal dependence remain explicit.
- Spread/error association is not calibration.
- Historical and replay-equivalent training costs are not interchangeable unless the validation gate passes.
- Different adopted checkpoints were not trained under a matched optimization budget.
- High-N dummy-query results measure throughput and memory only, not accuracy or physically validated super-resolution.
- Hardware, precision, chunking, cache policy, solver, and method-native training/inference settings limit absolute cost comparisons.
- No NFE/solver main panel and no A0/A1/A2/A3 ablation training are part of V4.

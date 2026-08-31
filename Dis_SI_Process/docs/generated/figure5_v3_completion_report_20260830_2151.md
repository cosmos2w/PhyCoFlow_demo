# Figure 5 V3 completion report

- Generated: `20260830_2151`
- Status: **Complete: both formal V3 runs and the strict figure build passed.**
- Formal UQ run: `uq_compare_formal_20260830_v3r6`
- Formal clean-cost run: `formal_cost_clean_v3_20260830_v3`
- Pilot runs: `pilot_uq_compare_20260830_v3, pilot_uq_compare_deterministic_20260830_v3, pilot_uq_compare_common_norm_20260830_v3, pilot_uq_compare_prepared_20260830_v3r2, pilot_uq_compare_prepared_20260830_v3r3, pilot_uq_compare_prepared_20260830_v3r4`

## Provenance and supersession

V2 outputs remain unchanged as provenance. The V2 DMF median of 127.05 ms from `formal_cost_20260830_v2` is superseded because its timing boundary included generic adapter/host-transfer overhead and was vulnerable to shared-GPU contamination. It must not be used as manuscript evidence. The V3 clean benchmark excludes loading, data I/O, CPU preprocessing, host transfers, generic dispatch, metrics, output transfer, and disk I/O while retaining required noise generation, value-dependent conditioning, model evaluations, observation consistency, and device-side output.

DMF chunk profiling resolved the provisional timing discrepancy. The adopted configurations specify an 8,192-point reconstruction chunk; V3 uses that canonical setting and permits only reusable static geometry. The prior approximately 29 ms result is consistent within 20% with the profiled 4,096-point streaming boundary, but is not the promoted coordinate.

The first five-method 12×8 pilot found same-seed drift only for Latent FM under nondeterministic CUDA execution (approximately 0.004–0.005 normalized max absolute difference). Deterministic and prepared-path reruns passed all stochasticity, reproducibility, normalization, and exact-path-equivalence gates; all pilot IDs are retained. Existing matching DMF U2 fieldwise spread/error/reliability summaries were reused in place, while the missing normalized CRPS reducer required matching-seed DMF draws.

## Main findings and narrative checks

Lowest mean normalized CRPS: DMF-Gen (0.0667). Weakest spread/error association: Latent FM (ρ=-0.033); 95% intervals cross zero for FFM-FNO, Latent FM. Corrected DMF native latency: 16.69 ms; direct core: 16.76 ms; independent exact-shape reprobe: 16.68 ms. The historical approximately 29 ms probe maps to 24.60 ms under the earlier 4,096-point streaming chunk (relative difference 15.2%).

Variable-query curves: DMF-Gen, FFM-Perceiver, MLP-RBF, Senseiver. Native-only markers: FFM-FNO, Latent FM, SiT, Geo-FNO.

The provisional narrative is only partly supported: DMF-Gen has the lowest CRPS, but spread/error association is weak enough to cross zero for FFM-FNO and Latent FM, and the accuracy–latency measurements do not establish unqualified Pareto superiority.

## Scope

Full reliability/interval-width curves, fieldwise uncertainty, cold/no-cache timing, reserved memory, and NFE diagnostics remain SI/internal. The optional 100k–1M throughput-only extension was not run. No A0/A1/A2/A3 ablation training was started.

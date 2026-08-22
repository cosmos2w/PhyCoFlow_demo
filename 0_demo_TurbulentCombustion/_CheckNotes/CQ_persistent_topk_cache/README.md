# CQ persistent geometry-only Top-K cache

Date: 2026-08-22  
Feature commit: `f600d65`

## Outcome

Persistent Top-K geometry reuse is implemented in the existing cached-streamed GL-RBF path. It adds no KNN implementation and does not alter CQ architecture, training, RF mathematics, K=32, or KeOps semantics. The inference-only cache contains `topk_d2`, `topk_idx`, `topk_valid`, and lightweight validity metadata.

Recommendation: on fixed query and sensor geometry, build persistent geometry once and use `cache_level="static_features"`. At one million queries, cached Euler NFE=4 takes 0.363 s steady state, only 0.121 s more than NFE=2, performs zero new KNN searches, and is 9.02x faster than no caching. This makes NFE=4 the preferred CRPS/inference operating point when NFE=4 is scientifically preferred.

This improves inference efficiency only. CQ-LR is substantially faster/lighter than F0-ENH, but clean 1,000-epoch CQ-LR validation remains about 10% worse. Persistent caching does not solve that quality gap; F0 remains the formal 3-D quality reference.

## Pre-edit cache-path audit

1. The CQ class is `ConditionalPointHybridLocalGlobalRBFCQ`.
2. It inherits `ConditionalPointHybridLocalGlobalRBF` and reuses `_get_topk_neighbors`, `_knn_search_keops`, and `_knn_search_torch`.
3. `prepare_condition_context` creates sensor tokens, latents, global summary, refined sensor features, and sensor-importance bias once per sample call.
4. CQ adds `global_q`; CQ-LR also projects current-condition `cq_latent_k` and `cq_latent_v` once. These value-dependent tensors are never persistent.
5. `prepare_query_context(cache_level="none")` stores no Top-K state, so local gather calls KNN during each velocity evaluation.
6. `geometry` builds Top-K before the ODE loop and reuses it over NFE.
7. `static_features` builds current-condition local RBF and CQ query-global readout before the loop; neither executes in the loop.
8. `_sample_cached_streamed` creates one condition context and query context, then streams Euler/Heun chunks.

Pre-edit CQ-LR calls for 13 queries, wrapper chunk 5, gather chunk 4:

| Cache level | Euler 1 | Euler 2 | Euler 4 | Euler 8 | Heun 2 |
|---|---:|---:|---:|---:|---:|
| none | 5 | 10 | 20 | 40 | 20 |
| geometry | 3 | 3 | 3 | 3 | 3 |
| static_features | 3 | 3 | 3 | 3 | 3 |

Stage 4 already cached Top-K within one trajectory. The new feature only adds reuse across separate `sample()` calls.

## Public API and cache separation

```python
geometry_cache = model.prepare_reconstruction_geometry_cache(
    coords=coords, obs_coords=obs_coords, obs_mask=obs_mask, chunk_size=8192,
)
reconstruction = model.sample(
    coords=coords, obs_coords=obs_coords, obs_values=new_sensor_values,
    obs_mask=obs_mask, obs_field_ids=obs_field_ids, n_steps=4,
    reconstruction_execution_mode="cached_streamed",
    reconstruction_cache_level="static_features",
    reconstruction_geometry_cache=geometry_cache,
)
```

The cache remains reusable when sensor values, RF source draws, NFE, or solver change. Each call rebuilds condition context, refined features, sensor importance, CQ-LR latent K/V, current local static feature, and CQ readout. For `static_features`, existing `_aggregate_topk_from_geometry` combines persistent geometry with current refined features, learnable sigma, and sensor importance without KNN.

Validation rejects changes in query/sensor/mask shape, device, dtype, storage pointer, stride, storage offset, in-place tensor version, K, or gather mode. Cache tensor shape/device/dtype is checked. Persistent geometry with `legacy_full` or cache level `none` is rejected. No large checksum is computed. The implementation is reusable by CQ-LR, CQ-Full, CQ-160, and F0-ENH.

## Numerical and regression evidence

The dedicated suite covers CQ-LR Euler NFE 1/2/4 and Heun NFE 2 for persistent `geometry` and `static_features`; changing sensor values A/B; one CQ-Full case; stale query/sensor/mask tensors; in-place mutation; changed K/mode; geometry-only payload; legacy rejection; zero post-build KNN calls; and per-condition CQ-LR latent K/V projection.

- Persistent suite: 38 passed.
- CQ plus Stage-4 focused matrix: 90 passed.
- Complete maintained `tests/` suite: 116 passed.
- Production KeOps maximum persistent-versus-fresh absolute difference: `3.34e-6`.
- CQ-LR epoch-1000 SHA-256: `36a35d28e5d2a8434ea24cafadba214714cfc346a5d83b8477c14dab9a82fb14`.
- Immutable F0 `best.pt` SHA-256: `e93198bc2cba3f001024bbc9c1b197b2b56ecd52d8967bb38592ee5090e95569`.

After geometry construction, persistent geometry and static-feature calls make zero `_get_topk_neighbors` calls. CQ-LR latent K/V projection occurs once per condition call and twice across calls with two sensor-value sets.

## Production benchmark

Checkpoint: clean A/B CQ-LR DemoN9511 epoch 1000. Hardware: NVIDIA RTX 6000 Ada. FP32, batch 1, M=256, K=32, chunk=8192, Euler NFE 1/2/4/8, five repeats, one discarded per-mode warm-up, plus KeOps/model warm-up.

Modes are no cache, geometry per call, static features per call, persistent geometry, and persistent geometry plus static features.

### Geometry construction and memory

| Queries | Build (s) | Build calls | Geometry MiB | Static cache MiB | Peak allocated MiB | Peak reserved MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 250,000 | 0.0387 | 31 | 99.18 | 427.2 | 873.7 | 1,128.0 |
| 1,000,000 | 0.1535 | 123 | 396.73 | 1,709.0 | 3,448.2 | 4,356.0 |
| 1,953,125 (125 cubed) | 0.2940 | 239 | 774.86 | 3,337.9 | 6,722.3 | 8,464.0 |

Geometry is exactly 13 bytes per query-neighbor pair: FP32 distance, int64 index, bool validity. Static-cache memory excludes separately retained persistent geometry; process peak includes all live tensors.

### One-million-query comparison

| NFE | Mode | Wall s | Top-K calls | Speedup vs none |
|---:|---|---:|---:|---:|
| 1 | none | 0.857 | 489 | 1.00x |
| 1 | geometry/call | 0.356 | 123 | 2.41x |
| 1 | static/call | 0.308 | 123 | 2.79x |
| 1 | persistent geometry | 0.202 | 0 | 4.25x |
| 1 | persistent plus static | 0.190 | 0 | 4.52x |
| 2 | none | 1.658 | 978 | 1.00x |
| 2 | geometry/call | 0.541 | 123 | 3.06x |
| 2 | static/call | 0.363 | 123 | 4.57x |
| 2 | persistent geometry | 0.372 | 0 | 4.46x |
| 2 | persistent plus static | 0.242 | 0 | 6.85x |
| 4 | none | 3.271 | 1,956 | 1.00x |
| 4 | geometry/call | 0.878 | 123 | 3.73x |
| 4 | static/call | 0.483 | 123 | 6.77x |
| 4 | persistent geometry | 0.707 | 0 | 4.63x |
| 4 | persistent plus static | 0.363 | 0 | 9.02x |
| 8 | none | 6.524 | 3,912 | 1.00x |
| 8 | geometry/call | 1.532 | 123 | 4.26x |
| 8 | static/call | 0.718 | 123 | 9.09x |
| 8 | persistent geometry | 1.387 | 0 | 4.70x |
| 8 | persistent plus static | 0.597 | 0 | 10.93x |

At NFE=4, persistent static is 1.33x faster than per-call static and 2.42x faster than per-call geometry. Components are 0.0043 s condition, 0.1059 s query context, and 0.2395 s ODE loop. Per-call static query construction is 0.2254 s, so persistent geometry removes about 0.120 s of KNN work.

### Persistent-static NFE scaling

| Queries | NFE 1 | NFE 2 | NFE 4 | NFE 8 | NFE4 minus NFE2 | NFE4 vs none |
|---:|---:|---:|---:|---:|---:|---:|
| 250,000 | 0.051 | 0.066 | 0.097 | 0.155 | 0.031 | 8.63x |
| 1,000,000 | 0.190 | 0.242 | 0.363 | 0.597 | 0.121 | 9.02x |
| 1,953,125 | 0.351 | 0.468 | 0.697 | 1.144 | 0.229 | 9.22x |

Five-call amortized NFE=4 latency including one-fifth of geometry build is 0.105 s, 0.394 s, and 0.756 s at the three scales.

## Deployment interpretation

Existing Stage-4 caching removes repeat KNN within a trajectory. Persistent geometry runs KNN once per fixed geometry session and zero times for later reconstructions, including new sensor values, solvers, NFE, and RF samples. For CFD/geothermal meshes and CRPS ensembles, build geometry outside the update/ensemble loop and use `static_features`; never persist condition context or CQ-LR latent K/V.

At one million queries, NFE=4 adds only 0.121 s over NFE=2 while remaining 9.02x faster than no cache. NFE=4 is therefore practical as the default when its established CRPS benefit is desired. NFE=8 is inexpensive but has no quality requirement in this task.

## Evidence files

- `benchmarks/cq_lr_250k.csv` and `.json`
- `benchmarks/cq_lr_1m.csv` and `.json`
- `benchmarks/cq_lr_125cubed.csv` and `.json`
- `benchmarks/summary.json`
- `smoke/results.csv` and `.json`
- `logs/` raw benchmark stdout

The optional masked-attention diagnostic was not run because it is independent and no attention change is included.

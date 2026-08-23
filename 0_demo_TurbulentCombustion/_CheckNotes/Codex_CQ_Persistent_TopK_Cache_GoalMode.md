# Parallel Goal-Mode Task — Persistent Geometry-Only Top-K Cache for GL_rbf_ENH_CQ

## Context

Repository: `cosmos2w/PhyCoFlow_demo`

This task targets the **current local Stage-6 CQ implementation**, especially:

- `GL_rbf_ENH_CQ`
- CQ-LR: query width 128, rank-64, four-head compact latent readout
- existing Stage-4/5 cached-streamed reconstruction/context APIs

Do not work from an older pre-CQ snapshot. Inspect the current local CQ implementation first and use its real class names/signatures.

The current clean A/B result is:

| Metric | F0-ENH | CQ-LR | CQ-LR change |
|---|---:|---:|---:|
| Mean epoch time | 27.150 s | 18.294 s | -32.62% |
| Diagnostic train step | 581.410 ms | 379.276 ms | -34.77% |
| Peak allocated | 27,642.9 MiB | 23,258.3 MiB | -15.86% |
| Peak reserved | 36,414.0 MiB | 27,688.0 MiB | -23.96% |
| Best validation loss | 0.353095 | 0.388921 | +10.15% |
| Final validation loss | 0.361207 | 0.400808 | +10.96% |

Interpretation:

- CQ-LR is a clear computational-efficiency improvement.
- CQ-LR is **not yet promoted as the scientific replacement for F0** because the ~10% validation regression is materially larger than the intended Stage-6 tolerance.
- Persistent Top-K caching is orthogonal to this quality issue and should benefit CQ-LR, CQ-Full, CQ-160, or F0.

Do not change CQ architecture or train a new model in this task.

---

# Goal

Verify and implement an explicit **persistent geometry-only Top-K cache** for the current CQ reconstruction path.

The intended deployment scenario is repeated inference on:

- the same reconstruction mesh/query coordinates;
- the same physical sensor locations and validity mask;
- changing sensor values and/or repeated stochastic reconstruction requests;
- possibly different NFE or solver choices.

Top-K geometry depends only on:

```text
query coordinates
sensor coordinates
sensor validity mask
K / gather mode
```

It does not depend on:

```text
sensor values
latent features
current RF state x_t
RF time t
RFF source draw
NFE
Euler vs Heun
```

Therefore geometry should be buildable once and reused across repeated `sample()` calls.

---

# 1. Audit the current CQ cache path before editing

Read the actual local CQ code and document:

1. CQ backbone class name.
2. Whether it reuses/inherits the old GL-RBF `_get_topk_neighbors`.
3. Current `prepare_condition_context`.
4. Current `prepare_query_context`.
5. Current `forward_query_chunk`.
6. Current `PointCloudFFM._sample_cached_streamed`.
7. Current reconstruction cache levels.
8. CQ-LR latent K/V caching behavior.

Confirm whether Stage-4 behavior remains:

## cache_level="none"

Top-K search is executed from the normal local-gather path during velocity evaluation.

## cache_level="geometry"

`topk_d2`, `topk_idx`, and `topk_valid` are built before the ODE loop and reused over NFE.

## cache_level="static_features"

The local RBF condition is built before the ODE loop. For CQ-LR, inspect exactly which latent/query readout terms are also cached.

If current CQ already caches geometry inside a single reconstruction trajectory, do not duplicate that work.

The new feature is **cross-call persistent geometry reuse**.

Record this audit in:

`_CheckNotes/CQ_persistent_topk_cache/README.md`

---

# 2. Scope constraint

Do not use this task to fix the CQ-LR validation gap.

Do not change:

- `cq_query_dim`;
- CQ readout rank/heads;
- CQ fusion;
- CQ-Full/CQ-LR architecture;
- latent architecture;
- sensor architecture;
- K=32;
- KeOps semantics;
- learnable RBF sigma;
- RF objective;
- training schedule.

This task is inference-only execution engineering.

---

# 3. Persistent geometry cache API

Add a reusable geometry-only cache API to the CQ-capable backbone/wrapper.

Preferred public wrapper API:

```python
geometry_cache = model.prepare_reconstruction_geometry_cache(
    coords=coords,
    obs_coords=obs_coords,
    obs_mask=obs_mask,
    chunk_size=8192,
)

recon = model.sample(
    coords=coords,
    obs_coords=obs_coords,
    obs_values=new_values,
    obs_mask=obs_mask,
    obs_field_ids=obs_field_ids,
    ...,
    reconstruction_execution_mode="cached_streamed",
    reconstruction_cache_level="geometry",  # or static_features
    reconstruction_geometry_cache=geometry_cache,
)
```

Backbone-side concept:

```python
prepare_query_geometry(
    coords,
    obs_coords,
    obs_mask,
    chunk_size,
)
```

The cache contains only:

```text
topk_d2
topk_idx
topk_valid
geometry metadata
```

Do not include:

```text
refined_sensor_feat
sensor importance
latent state
local_cond
CQ-LR latent K/V
query-global readout
obs_values
x_t
```

Those are condition/model-dependent, not geometry-only.

---

# 4. Keep CQ-LR cache layers separate

CQ-LR has additional condition/readout state compared with F0.

Keep these layers conceptually separate:

## Persistent geometry cache

May persist across calls when mesh/sensor locations are unchanged.

## Per-call condition context

Must be recomputed when `obs_values` change because sensor tokens, latents, refined sensor features and CQ-LR latent K/V depend on measurement values.

## Per-call static feature cache

May reuse persistent geometry while rebuilding local RBF features and CQ latent readout for the current values.

Do not accidentally persist CQ-LR latent K/V across calls with changed observation values.

---

# 5. Safe invalidation

A stale geometry cache is unacceptable.

Validate at least:

- query shape;
- sensor-coordinate shape;
- mask shape;
- device;
- coordinate dtype;
- K;
- gather mode;
- query tensor storage pointer;
- sensor-coordinate tensor storage pointer;
- sensor-mask storage pointer.

The simplest safe implementation may require the same geometry tensors to remain alive between calls. That is fine.

Do not compute expensive million-point checksums in the hot path.

Sensor values must not be used for invalidation.

Changing sensor values while keeping geometry fixed must allow reuse.

Changing query geometry, sensor positions, validity mask, K, or gather mode must fail clearly.

---

# 6. Use the supplied helper module

Use `persistent_topk_geometry_cache.py`.

It provides:

- a small `PersistentTopKGeometryCache` dataclass;
- geometry construction through the backbone's existing `_get_topk_neighbors`;
- validation;
- byte accounting.

Adapt only if the current CQ implementation renamed the neighbor-search primitive.

Do not duplicate a second KNN implementation.

The helper must call the exact existing CQ/F0 neighbor backend so KeOps/Torch behavior remains unchanged.

---

# 7. Integrate with `prepare_query_context`

Extend the current CQ contract with something equivalent to:

```python
prepare_query_context(
    coords,
    condition_context,
    cache_level,
    chunk_size,
    precomputed_geometry=None,
)
```

Behavior:

### No precomputed geometry

Preserve current behavior.

### With precomputed geometry

Validate it and use:

```text
topk_d2
topk_idx
topk_valid
```

without calling KNN.

For `cache_level="geometry"`:

reuse those tensors directly.

For `cache_level="static_features"`:

use the existing geometry-based aggregation routine to build the current-condition local RBF feature from:

```text
persistent geometry
+
current refined sensor features
+
current learnable sigma
+
current sensor-importance bias
```

No KNN search.

CQ-LR latent readout should continue using its own current-condition cache path.

---

# 8. Integrate with `sample()`

Add an optional argument:

```python
reconstruction_geometry_cache=None
```

to CQ-capable `PointCloudFFM.sample()` and `_sample_cached_streamed()`.

Default `None` must preserve existing behavior.

Do not modify `legacy_full` semantics.

If geometry cache is supplied with a non-cached reconstruction mode, reject it clearly.

---

# 9. Verify current intra-trajectory caching on CQ

Before measuring the new cross-call feature, instrument actual Top-K search calls.

Use a call counter on the real CQ neighbor-search primitive.

Test:

```text
cache_level:
    none
    geometry
    static_features

Euler:
    NFE 1,2,4,8

Heun:
    NFE 2
```

Expected:

- `none`: Top-K calls grow with velocity evaluations/NFE.
- `geometry`: KNN calls independent of NFE.
- `static_features`: KNN calls independent of NFE and local RBF is absent from the ODE loop.

If this is not true in current CQ, fix the intra-trajectory regression first.

---

# 10. Numerical tests

Use `test_cq_persistent_topk_cache.py`.

Adapt only the CQ model-construction helper to the actual local class/factory.

Mandatory cases:

## A. Fresh geometry vs persistent geometry

For identical CQ weights and inputs:

```text
fresh per-call geometry
vs
persistent precomputed geometry
```

must agree tightly for:

- CQ-LR;
- `topk_rbf_glres`;
- Euler NFE 1/2/4;
- Heun NFE 2.

If CQ-Full remains available, also test at least one CQ-Full case.

## B. Change sensor values

Build geometry once.

Run values A and B.

For each values set compare fresh geometry vs the same persistent geometry. They must match.

## C. Invalid geometry

Replacing query coords, sensor coords, mask, or K must trigger validation failure.

## D. Zero KNN calls after build

After `prepare_reconstruction_geometry_cache()`, a `sample()` call using it must perform **zero** new `_get_topk_neighbors` calls.

---

# 11. Benchmark

Use `benchmark_cq_persistent_topk_cache.py`.

Adapt its CQ model factory to the actual local CQ code/config.

Primary model/checkpoint:

- CQ-LR 1000-epoch checkpoint from the clean Stage-6 A/B package.

Optionally benchmark F0 too, but CQ-LR is primary.

Query sizes:

```text
250,000
1,000,000
1,953,125  # 125^3
```

Settings:

```text
batch=1
M=256
K=32
chunk=8192
NFE=1,2,4,8
repeats=5
Euler
```

Compare:

```text
A. cached_streamed + cache_level=none
B. cached_streamed + geometry built inside every sample call
C. cached_streamed + static_features built inside every sample call
D. cached_streamed + persistent geometry + cache_level=geometry
E. cached_streamed + persistent geometry + cache_level=static_features
```

Record:

```text
geometry_build_s
sample_wall_s
amortized_wall_s
topk_calls
peak_allocated_mb
peak_reserved_mb
geometry_cache_mb
query/static cache_mb
speedup_vs_none
speedup_vs_per-call geometry
```

If practical split condition-context, query-context and ODE-loop times.

---

# 12. Memory accounting

Persistent geometry is not free.

For FP32 `d2` + int64 indices + bool valid mask, storage is about:

```text
13 bytes per query-neighbor pair
```

At K=32:

```text
1M queries       ~397 MiB
125^3 queries    ~775 MiB
```

Report measured values.

Do not change index dtype or distance precision in this task.

---

# 13. Deployment / NFE interpretation

Report two speedups separately:

## Existing intra-trajectory caching

KNN once per reconstruction instead of once per NFE.

## New cross-call persistent caching

KNN once for the geometry/session and zero times on subsequent reconstructions.

The second is especially valuable for:

- fixed CFD/geothermal meshes;
- fixed physical sensor positions;
- repeated updates as sensor values change;
- CRPS/ensemble generation;
- repeated NFE studies.

For CRPS/ensemble inference, build geometry outside the ensemble loop.

Because NFE=4 reportedly gives the best CRPS, compare NFE 1/2/4/8 after geometry is already persistent.

Report marginal cost:

```text
T(NFE=4) - T(NFE=2)
```

The scientific question is whether persistent geometry makes NFE=4 cheap enough to become the deployment default.

---

# 14. Training is not part of this task

Do not cache Top-K across optimizer steps.

CQ training uses changing query/sensor layouts and one RF time per update.

Persistent geometry is an inference optimization.

---

# 15. Optional masked-attention diagnostic

Run only as a separate diagnostic.

On actual H100/BF16 CQ shapes compare:

- current `nn.MultiheadAttention` boolean `key_padding_mask`;
- equivalent additive/SDPA mask;
- direct `scaled_dot_product_attention` if appropriate.

Measure forward/backward separately and verify outputs/gradients.

Only propose a separate production patch if the gain is reproduced and mathematically equivalent.

Do not mix attention changes into the Top-K cache commit.

---

# 16. Do not hide CQ quality status

The final README must state:

- CQ-LR is substantially faster/lighter than F0;
- current clean 1000-epoch CQ-LR validation is still ~10% worse;
- persistent Top-K caching improves inference efficiency only and does not solve that quality gap.

---

# 17. Success criteria

Pass when:

1. current CQ intra-trajectory Top-K behavior is verified;
2. persistent geometry works with CQ-LR;
3. cache is independent of sensor values;
4. repeated sample calls perform zero new KNN searches after build;
5. Euler/Heun match fresh-cache reference;
6. largest feasible benchmark shows strong repeated-inference gain;
7. NFE scaling is measured;
8. CQ architecture/checkpoint values are unchanged;
9. regression suite passes;
10. diff is small and reusable by CQ-Full/CQ-160/F0.

---

# 18. Deliverables

Create:

`_CheckNotes/CQ_persistent_topk_cache/README.md`

plus raw CSV/JSON.

Report:

- current CQ class/context API audit;
- commit SHA;
- files changed;
- Top-K call-count matrix;
- numerical equivalence;
- geometry-cache memory;
- first-call vs repeated-call latency;
- NFE 1/2/4/8;
- 125^3 results if feasible;
- CRPS/NFE deployment implication;
- optional masked-attention result;
- any changes required in CQ helper/factory code.

Keep persistent geometry caching as a separate, minimal inference feature.

# PointCloud FFM Efficiency — Codex Goal-Mode Plan for Stages 1–5

## Working context

Repository:

`cosmos2w/PhyCoFlow_demo`

Required branch:

`perf/pointcloud-ffm-field-reconstruction`

Current baseline commit to build from:

`0e5854e` — `perf(pointcloud): optimize FFM training data path`

Primary working area:

- `0_demo_TurbulentCombustion/src/train_pointcloud_ffm.py`
- `0_demo_TurbulentCombustion/src/helpers.py`
- `0_demo_TurbulentCombustion/src/pointcloud_data_path.py`
- `0_demo_TurbulentCombustion/src/Model.py`
- `0_demo_TurbulentCombustion/Save_config/config_pointcloud_ffm.yaml`
- `0_demo_TurbulentCombustion/tests/`
- `0_demo_TurbulentCombustion/_CheckNotes/`

The current optimized standard-training path has already shown approximately:

- 11.3% lower steady training time,
- 12.7% higher throughput,
- 57% lower host RSS,
- 35% lower CPU time,
- essentially unchanged final training loss,
- nearly unchanged peak GPU training memory because model activations now dominate,
- and `indexed_union` HDF5 reads are pathological on the current contiguous HDF5 layout.

Do **not** undo those changes.

---

# Goal

Complete Stages 1–5 of the PointCloud FFM efficiency work before any Stage-6 architecture modification.

The core goal is:

> Preserve the current mathematical PointCloud FFM / GL-RBF model and Rectified-Flow objective, while making training supervision and full-field reconstruction scale much better to large 3-D meshes.

This is still an experimental performance branch. Keep legacy/reference behavior available where it is useful for controlled A/B checks, but do not let legacy compatibility prevent a clean optimized execution path.

After Stages 1–5 are complete and validated, prepare a **limited validation run only**. Do not launch a long formal production training campaign in this task.

Stage 6 — changing the model architecture itself — is **out of scope**.

Examples of Stage-6 work that must not be performed here:

- changing `hidden_dim`, `cond_dim`, or `latent_dim` as a model redesign;
- reducing latent count/heads/blocks as the main solution;
- removing query latent readout as an architectural choice;
- replacing `topk_rbf_glres` with another architecture;
- changing the GL-RBF fusion formulation;
- introducing anchor/hierarchical decoders;
- changing RF mathematics or the loss definition.

Stages 4–5 are allowed to refactor **execution APIs** around the existing modules as long as the same model function and gradients are preserved.

---

# Goal-mode working rules

Work continuously through the stages in order.

At each stage:

1. inspect the existing implementation before editing;
2. make the smallest coherent implementation that achieves the stage goal;
3. add focused tests;
4. run the focused tests;
5. record benchmark/correctness evidence under `_CheckNotes/`;
6. only proceed when the stage gate passes.

Do not ask for confirmation between stages unless genuinely blocked by missing data/files or an unrecoverable ambiguity.

Prefer one clean commit per major stage, or at minimum leave the branch in a bisectable state with clear commit messages.

Do not rewrite or force-push existing history.

Keep a progress log, for example:

`_CheckNotes/Stage1_5_Efficiency_GoalMode.md`

with:

- stage status,
- files changed,
- tests run,
- measured results,
- unresolved limitations,
- exact commands used.

---

# Hard invariants

The following must remain unchanged through Stages 1–5 unless explicitly stated otherwise:

- GL-RBF parameterization and learned weights;
- `GL_rbf_ENH` mathematical forward mapping;
- `topk_rbf` / `topk_rbf_glres` gather mathematics;
- KeOps neighbor semantics;
- Rectified Flow convention;
- RFF prior distribution;
- training target `x1 - x0`;
- loss definition;
- observation semantics;
- solver semantics for Euler and Heun;
- sensor consistency semantics;
- checkpoint compatibility where practical.

The old full execution path should remain callable during validation so exact/near-exact numerical comparisons can be made against the optimized execution path.

Do not silently switch existing checkpoints to a different model.

---

# Stage 1 — Close Round 1 with matched evaluation and order-control timing

## Objective

Resolve the two remaining ambiguities from the first 100-epoch comparison:

1. the final validation difference was measured with different stochastic query/observation layouts;
2. the legacy run executed before the optimized run, so the optimized run may have benefited from filesystem cache warming.

No model changes are required in Stage 1.

---

## 1.1 Add fixed validation manifests

Create a reusable validation-manifest utility.

Suggested location:

`src/pointcloud_eval_manifest.py`

or another clearly named module.

A manifest should contain enough information to reproduce validation inputs exactly:

- dataset identifier/path;
- dataset/split fingerprint where practical;
- split name;
- snapshot/time indices;
- query indices for each validation sample;
- observation indices;
- observation field IDs;
- observation masks/counts;
- conditioned fields;
- `n_query_points`;
- sensor-count settings;
- sampling seed;
- manifest version;
- optionally a checksum/hash.

Store manifests in a simple durable format, preferably `.pt` plus a small human-readable `.json` summary.

The manifest must **not** contain model outputs.

Add deterministic generation and loading.

Given the same seed/config/dataset, manifest generation must reproduce the same indices.

---

## 1.2 Add matched checkpoint evaluation

Create a small script, for example:

`src/evaluate_pointcloud_fixed_manifest.py`

It should accept:

- one or more checkpoint paths;
- a fixed validation manifest;
- dataset/config path;
- device;
- number of repeated RF-loss evaluations if requested.

Important: current validation loss also samples RF time `t` and the RFF prior.

Therefore fixed query/observation indices alone are not sufficient for a fully controlled RF-loss comparison.

For matched validation-loss evaluation, also control the RF RNG stream.

Accept either of these implementations:

### Preferred simple implementation

Before each matched `training_loss()` call:

- restore/reset the same PyTorch CPU/CUDA RNG state;
- use the same batch;
- call the same unchanged `training_loss()`.

Because the active model uses zero dropout in the current configuration, this should yield identical `t` and RFF source draws between checkpoints for the same batch.

### Alternative

Add a diagnostic-only wrapper that explicitly supplies fixed `t` and fixed RFF source draws without changing the normal training API.

Do not change the default training objective.

Report for each checkpoint:

- mean RF validation loss;
- standard deviation across manifests/repetitions;
- paired per-manifest differences.

If the existing Round-1 checkpoints are available under `_CheckNotes/Round1_runs/`, evaluate them. If not present in the checkout, leave the script ready and document the exact command.

---

## 1.3 Add short reversed-order timing control

Prepare a short timing comparison in reverse order:

1. optimized first;
2. legacy second.

Do not repeat another 100-epoch formal run.

A 10–20 epoch timing-only control is sufficient.

Use:

- same physical GPU;
- same dataset;
- same model;
- same batch/query/observation settings;
- reconstruction disabled;
- diagnostics disabled;
- same logging interval for both paths;
- no loss plotting;
- minimal validation.

Record steady epoch time after warm-up.

The purpose is only to check whether the approximately 10% training-time advantage survives reversing run order.

---

## Stage-1 gate

Stage 1 passes when:

- fixed manifests are reproducible;
- matched tensor inputs are identical;
- matched RF-loss evaluation uses controlled RNG;
- the existing checkpoints show no obvious systematic material regression under matched evaluation, or any difference is documented quantitatively;
- short reversed-order timing still shows a meaningful optimized-path advantage;
- no model code/math was changed.

Do not block later stages solely because the two existing checkpoints have slightly different validation losses if the matched evaluation shows the difference is within stochastic variability.

---

# Stage 2 — Final data-path polish for million-point scaling

## Objective

Remove remaining low-risk full-mesh CPU work and long-run diagnostic overhead while retaining explicit old behavior for A/B checks.

This stage still does not change the model.

---

## 2.1 Separate field read mode from normalization mode

The current optimized `legacy_full_snapshot` path still does:

```text
read complete raw snapshot
-> normalize complete snapshot
-> gather query and observation points
```

Add an explicit normalization/materialization option.

Suggested config:

```yaml
field_normalization_mode: "selected_after_full_read"
# choices:
#   "legacy_full_after_read"
#   "selected_after_full_read"
```

Semantics:

### `legacy_full_after_read`

Preserve current behavior exactly:

```text
raw_full
-> normalize full [N_full, C]
-> gather selected values
```

### `selected_after_full_read`

Use one sequential full snapshot read, but normalize only the union of required points:

```text
raw_full
-> union(query_indices, obs_indices)
-> gather raw selected rows
-> normalize selected rows
-> map back to query/obs tensors
```

This keeps the fast sequential HDF5 read while removing unnecessary full-field normalization and the full normalized CPU tensor.

Prefer one deduplicated/sorted union to avoid normalizing query/observation overlaps twice.

Correctness target:

Selected tensors must agree with `legacy_full_after_read` within `rtol=atol=1e-6`.

Keep `indexed_union` available as an experimental field-read option, but do not make it the default for the current contiguous HDF5 dataset.

---

## 2.2 Make diagnostics scalable to long runs

Current diagnostics retain all rows and rewrite full CSV/JSON state repeatedly.

Add a scalable storage mode.

Suggested config:

```yaml
data_path_diag_storage_mode: "append"
# choices:
#   "legacy_rewrite"
#   "append"
```

For `append`:

- append new CSV rows;
- optionally use JSONL for row-wise persistence;
- keep only the current/recent epoch window in memory;
- generate compact summary JSON periodically or at clean shutdown.

The diagnostic system must not become O(number_of_epochs²) in file-writing work.

Preserve the current behavior as `legacy_rewrite` until this branch is finalized.

---

## 2.3 Move `zero_grad(set_to_none=True)` before standard-training forward

In the standard training loop, move:

```python
optimizer.zero_grad(set_to_none=True)
```

before the model forward.

This is mathematically equivalent because standard training does not intentionally accumulate gradients across batches.

For temporary A/B verification, it is acceptable to add:

```yaml
zero_grad_placement: "before_forward"
# ["legacy_after_forward", "before_forward"]
```

but do not keep this option permanently unless it is useful.

Measure:

- allocated memory before forward;
- peak memory;
- step time.

Do not expect a large speedup; this is primarily a memory-lifetime cleanup.

Direct-coherence mode may keep its existing specialized update sequence unless a safe equivalent change is obvious.

---

## 2.4 Keep optimized standard-training defaults explicit

Recommended candidate after Stage 2:

```yaml
data_path_mode: "optimized"
coord_batch_mode: "shared_mesh"
index_sampling_mode: "scalable"
sampling_device: "cpu"
field_read_mode: "legacy_full_snapshot"
field_normalization_mode: "selected_after_full_read"
gpu_transfer_mode: "selected_only"

dataloader_persistent_workers: true
dataloader_prefetch_factor: 2
non_blocking_transfer: true

data_path_diag_storage_mode: "append"
```

Do not enable `indexed_union` by default on the current dataset.

---

## Stage-2 gate

Stage 2 passes when:

- selected-after-full-read matches old selected tensors;
- CPU normalization/materialization time and host memory do not regress;
- long-run diagnostics no longer rewrite an ever-growing history each epoch;
- standard training still passes regression tests;
- model/checkpoint math remains untouched.

---

# Stage 3 — Build a true 3-D / million-point scaling diagnostic

## Objective

Before changing model structure, quantify separately how runtime and memory scale with:

- full mesh size `N_full`;
- supervised query count `N_query`;
- observation count;
- batch size;
- reconstruction query count.

The key goal is to distinguish:

```text
full-mesh data cost
```

from:

```text
query-model cost
```

on large problem sizes.

---

## 3.1 Add a scaling benchmark utility

Suggested file:

`src/benchmark_pointcloud_scaling.py`

Support two benchmark classes.

### A. Data-path scaling

Measure:

- full snapshot read;
- selected normalization;
- index sampling;
- collate/materialization;
- H2D;
- host RSS where practical;
- selected GPU input memory.

Use the current real HDF5 dataset.

If a formal 3-D dataset path is supplied, support it without assuming a specific point count.

### B. Model execution scaling

Use GPU-resident or synthetic selected tensors to isolate:

- forward;
- backward;
- optimizer step;
- peak allocated/reserved memory.

This mode does not need physical correctness of the synthetic fields; it is a performance scaling test.

Use 3-D coordinates.

Do not write huge synthetic HDF5 files by default.

Optional explicit flags may create temporary HDF5 datasets for I/O-layout experiments, but they must not be part of the default benchmark.

---

## 3.2 Sweep `N_full` independently from `N_query`

Suggested default performance sweep:

```text
N_full:
    ~40k
    250k
    1M

N_query:
    4,096
    16,384
    65,536
```

Optionally include 131,072 if memory allows.

For synthetic model-only tests, `N_full` matters only to geometry/data preparation; model cost should be reported against `N_query`.

Use batch sizes that avoid intentional OOM.

Do not force `batch_size=144` at million-point stress sizes.

Record the actual batch size in every row.

---

## 3.3 Observation-count sweep

At minimum test:

```text
M = 256
M = 512
M = 1024
```

where feasible.

This is important because exact top-K search still scales approximately with `N_query × M` even with KeOps memory savings.

Do not modify KNN/gather mathematics in this stage.

---

## 3.4 Produce scaling tables

Save CSV/JSON with at least:

```text
N_full
N_query
N_obs
batch_size

read_ms
normalize_ms
index_sampling_ms
h2d_ms
pre_model_ms

forward_ms
backward_ms
optimizer_ms
step_ms

gpu_peak_allocated_mb
gpu_peak_reserved_mb
host_rss_mb if available
```

Also calculate derived metrics:

```text
queries / second
samples / second
milliseconds per 1k queries
GPU memory per 1k queries
```

Use diagnostic CUDA synchronization only on benchmark iterations.

---

## 3.5 Optional storage-layout experiment

Because `indexed_union` was pathological on the current contiguous HDF5 file, do not assume that point-chunking automatically solves random query access.

If easy to implement, add an **optional** small layout experiment:

- contiguous;
- point-chunked;
- time/snapshot-oriented chunked.

Test random uniform queries.

This is diagnostic only and must not delay the main stage.

The output should help decide how future formal 3-D datasets should be stored.

---

## Stage-3 gate

Stage 3 passes when there is a clear measured scaling report showing:

- how much residual standard-training time depends on `N_full`;
- how model time grows with `N_query`;
- how observation count affects the current GL-RBF path;
- whether data handling or model execution dominates at 250k–1M-point scale.

This report should explicitly identify the point where `N_query`, not the data path, becomes dominant.

---

# Stage 4 — Reconstruction execution refactor: static caching + end-to-end query streaming

## Objective

Make full-field reconstruction on very large meshes memory-safe and substantially cheaper **without changing the trained model function**.

The existing forward path recomputes static observation-conditioned work at every ODE function evaluation and builds large full-field hidden tensors.

Stage 4 should separate static and dynamic computation and stream complete query evaluation in chunks.

This is an execution refactor, not an architecture redesign.

---

## 4.1 Preserve a legacy reconstruction path

Add an explicit execution option.

Suggested config/API:

```yaml
reconstruction_execution_mode: "cached_streamed"
# choices:
#   "legacy_full"
#   "cached_streamed"
```

`legacy_full` must retain the current behavior for numerical A/B testing.

Do not remove the old forward path yet.

---

## 4.2 Refactor GL-RBF execution into static and dynamic pieces

For `ConditionalPointHybridLocalGlobalRBF`, introduce conceptually equivalent APIs such as:

```python
condition_ctx = model.prepare_condition_context(
    obs_coords,
    obs_values,
    obs_mask,
    obs_field_ids,
)

query_ctx = model.prepare_query_context(
    coords,
    condition_ctx,
    cache_level=...,
)

v_chunk = model.forward_query_chunk(
    t,
    x_t_chunk,
    coords_chunk,
    condition_ctx,
    query_ctx_chunk,
)
```

Exact names can differ.

The ordinary public `forward(...)` must continue to work.

Ideally implement `forward(...)` using the new primitives, or keep the legacy path side-by-side until equivalence is proven.

---

## 4.3 Identify static condition terms correctly

For the current GL-RBF model, the following are static over an ODE trajectory:

- sensor coordinate encoding;
- field embeddings;
- sensor tokens;
- latent input attention;
- latent self-attention;
- sensor-to-latent re-injection;
- global latent summary;
- sensor readback;
- refined sensor features;
- sensor-importance bias for `topk_rbf_glres`.

These depend on observations/model parameters but not on `x_t` or ODE time.

Compute them once per reconstruction condition.

---

## 4.4 Identify query-static terms carefully

Potentially static terms include:

- coordinate Fourier features;
- top-K neighbor indices;
- top-K squared distances;
- query-to-latent readout when `query_readout_type == "coord"`;
- local RBF condition when model parameters are frozen during inference.

Important distinctions:

### In inference

Model parameters and RBF sigma are fixed, so fully cached local/query-global features are valid.

### In training / Stage 5

Do **not** detach or freeze learnable quantities.

For training, prefer caching geometry (`topk_idx`, distances) rather than detached local features if that would break gradients to:

- sensor features;
- learnable RBF sigma;
- query readout modules.

Design the context API so it can be used in both inference and training without accidentally cutting gradients.

---

## 4.5 Add cache levels

Useful explicit options:

```yaml
reconstruction_cache_level: "static_features"
# choices:
#   "none"
#   "geometry"
#   "static_features"
```

Semantics:

### `none`

Only condition context is reused; query-static work is recomputed per chunk/NFE.

### `geometry`

Cache:

- query coordinate features if useful;
- KNN indices;
- KNN distances.

Recompute local weighted features as needed.

### `static_features`

Inference-only/full cache:

- local condition features;
- coordinate-based query latent readout;
- other static query-conditioned features.

Use BF16/FP16 cache only if added later as an explicit separate experiment. Do not introduce mixed precision silently in this stage.

Default numerical-validation path should remain FP32.

---

## 4.6 Implement end-to-end query streaming

Do not only chunk KNN and attention and then concatenate full hidden tensors.

The entire per-query velocity computation must happen inside one chunk loop.

Conceptually:

```python
for start, end in query_chunks:
    coords_c = coords[:, start:end]
    state_c = state[:, start:end]

    point_feat_c = ...
    global_c = ...
    local_c = ...
    velocity_c = final_head(...)

    update state[:, start:end]
```

Do not create full-field tensors like:

```text
[B, N, 256] point_feat
[B, N, 256] query_global
[B, N, 128] local_cond
[B, N, 640] head_in
```

during reconstruction.

The full state `[B, N, C]` is acceptable because it is the actual field and is small relative to hidden features.

---

## 4.7 Euler and Heun must preserve semantics

### Euler

Chunkwise update is straightforward because the existing query model is pointwise in the dynamic state once the shared observation context is fixed.

### Heun

For each chunk:

1. compute `v0_chunk`;
2. form `x_euler_chunk`;
3. compute `v1_chunk`;
4. apply the same Heun update.

Do not change the solver formula.

---

## 4.8 Observation consistency must remain equivalent

Test:

- `none`;
- `default_hard`;
- `endpoint`;
- `endpoint_smooth`.

If full pointwise/smooth consistency maps are created once, slice them by query chunk.

Do not change sensor-clamping semantics.

---

## 4.9 Reconstruction equivalence tests

For a fixed model/checkpoint and fixed sparse condition:

1. reset RNG to the same state;
2. run `legacy_full`;
3. reset RNG again;
4. run `cached_streamed`.

Test:

```text
gather_mode:
    topk_rbf
    topk_rbf_glres

solver:
    Euler
    Heun

NFE:
    1
    2
    4
```

At least on a small deterministic test case.

Use tight FP32 tolerances.

The goal is numerical equivalence, not merely similar metrics.

Also compare reconstruction metrics on one real validation snapshot.

---

## 4.10 Reconstruction scaling benchmark

Measure at:

```text
N = 40k
N = 250k
N = 1M
```

for batch size 1 and a small NFE such as 2.

Report:

- wall time;
- seconds per million points per NFE;
- peak allocated/reserved GPU memory;
- cache memory;
- legacy vs cached-streamed where legacy fits.

The optimized path should show that peak hidden-feature memory is controlled primarily by chunk size rather than total `N`.

---

## Stage-4 gate

Stage 4 passes when:

- cached-streamed FP32 reconstruction matches legacy within tight tolerance;
- Euler/Heun and observation consistency remain correct;
- a full large synthetic/real query field can reconstruct without allocating full hidden feature fields;
- peak memory scales mostly with chunk size plus final state/cache;
- static observation encoding is not recomputed every NFE.

---

# Stage 5 — Large-effective-query training with query microbatching

## Objective

Solve the main supervision-vs-memory problem:

> Allow one physical training condition to receive a much larger effective query set without putting all query activations in one autograd graph at the same time.

This stage must preserve the same RF objective.

`n_query_points` should represent the **effective total supervised query count per condition**.

Add a separate execution chunk size.

Suggested config:

```yaml
train_query_microbatch_size: null
# null or >= n_query_points -> historical single-query-batch behavior
# smaller value -> query microbatching

reuse_condition_context_across_query_microbatches: true
```

Do not confuse this with DataLoader batch size.

---

## 5.1 Refactor RF training wrapper without changing the objective

Current `training_loss()` internally:

1. samples RFF source `x0`;
2. samples one `t` per physical sample;
3. creates `x_t`;
4. creates target `x1 - x0`;
5. runs model;
6. computes MSE.

For microbatching, the RF stochastic state must be sampled once for the **whole effective query set**, not independently per query chunk.

A safe implementation is:

```text
sample all selected coords/targets for N_query_effective
sample x0 once on all selected coords
sample t once per physical sample

then iterate query chunks:
    x_t_chunk = ...
    target_chunk = ...
    predict velocity chunk
    weighted chunk loss
```

The selected full `x0` tensor is acceptable; it is only `[B, N_query, C]` and is much smaller than model hidden activations.

Do not call the old `training_loss()` independently for each query chunk because that would resample RFF coefficients and time and change the objective.

Instead, refactor the wrapper into reusable objective primitives, for example:

```python
rf_state = model.prepare_training_bridge(x1, coords)

loss_chunk = model.training_loss_chunk(
    rf_state,
    chunk_slice,
    condition_context=...
)
```

Exact API is flexible.

The existing `training_loss()` should still behave exactly as before when no microbatching is requested.

---

## 5.2 Preserve one coherent RFF source field

The current RFF prior draws one set of random Fourier coefficients per sample/channel for the entire coordinate set.

Microbatching must preserve that.

Preferred first implementation:

- sample the entire selected `x0` once;
- slice it per query microbatch.

Do not separately call the RFF prior for each query chunk.

---

## 5.3 Reuse observation-conditioned context

Use the Stage-4 condition-context API.

For training, the context must remain differentiable.

Do not detach it.

A straightforward initial implementation is acceptable:

- compute condition context once;
- process query microbatches sequentially;
- call backward on each weighted chunk;
- use `retain_graph=True` for all but the last chunk if required to reuse the shared context graph;
- call `optimizer.step()` only once after all chunks.

Delete chunk-local references promptly.

If another implementation can accumulate the context gradient more efficiently while remaining exactly equivalent, that is acceptable, but do not over-engineer before the simple correct path is measured.

---

## 5.4 Correct loss weighting

The existing MSE is a mean over all query entries.

For chunk `c` with `N_c` points out of `N_total`:

```text
weighted_loss_c = loss_c * (N_c / N_total)
```

assuming all chunks have the same channel count and valid-query semantics.

If masks are present, weight by the actual number of valid scalar elements instead.

The sum of weighted chunk losses must equal the monolithic loss up to numerical precision.

Do not average chunk means equally when the last chunk is smaller.

---

## 5.5 Correct optimizer sequence

For standard training:

```text
optimizer.zero_grad(set_to_none=True)

build differentiable condition context once
sample RF bridge once

for each query microbatch:
    compute weighted chunk loss
    backward

clip gradients once
optimizer.step() once
```

Do not clip gradients separately per query chunk.

Do not step the optimizer per chunk.

This must remain one training update per physical DataLoader batch.

---

## 5.6 Validation microbatching

Support the same query microbatch execution in validation with no gradients.

Accumulate the exact weighted total loss.

Validation should not require large hidden activation memory.

---

## 5.7 Gradient-equivalence tests

This is mandatory.

On a small fixed test problem:

1. initialize one model state;
2. create one fixed query/observation layout;
3. fix RNG;
4. run monolithic query training;
5. capture:
   - total loss;
   - all parameter gradients;
   - optimizer-updated parameters;
6. reload the same initial state;
7. reset RNG;
8. run query-microbatched training;
9. compare.

Test at least:

```text
effective queries: 31 or another non-divisible count
microbatch size: 7 or similar
```

so the final chunk has a different size.

Compare:

- loss;
- representative gradients;
- preferably all gradients;
- one optimizer update.

Use tight tolerances appropriate for FP32 summation order.

Also test with learnable RBF sigma so gradients are not accidentally lost.

---

## 5.8 Large-effective-query benchmark

Benchmark:

```text
effective N_query:
    4,096
    16,384
    65,536

microbatch:
    4,096
    8,192
```

where feasible.

Record:

- step time;
- forward/backward time;
- peak GPU memory;
- query throughput;
- condition-context time;
- query-chunk time.

The desired result is:

> Peak memory should be governed primarily by `train_query_microbatch_size`, not `n_query_points`.

Wall time may grow approximately linearly with total effective queries; that is expected.

The purpose is to preserve more supervision without OOM.

---

## Stage-5 gate

Stage 5 passes when:

- microbatched and monolithic losses agree;
- parameter gradients/one-step updates agree within FP32 tolerance;
- RFF source coherence is preserved across chunks;
- one optimizer step still corresponds to one physical batch;
- peak memory follows microbatch size;
- effective query counts much larger than 4,096 can train without reducing supervision solely for memory reasons.

---

# Integrated configuration after Stages 1–5

Do not delete the existing YAML options.

Add the new execution options clearly.

Suggested candidate section:

```yaml
# ==========================================================
# Stage 1–5 efficiency candidate
# ==========================================================

# ----- data path -----
data_path_mode: "optimized"
coord_batch_mode: "shared_mesh"
index_sampling_mode: "scalable"
sampling_device: "cpu"
field_read_mode: "legacy_full_snapshot"
field_normalization_mode: "selected_after_full_read"
gpu_transfer_mode: "selected_only"

dataloader_persistent_workers: true
dataloader_prefetch_factor: 2
non_blocking_transfer: true

data_path_diag_storage_mode: "append"

# ----- training query execution -----
n_query_points: 4096
train_query_microbatch_size: null
reuse_condition_context_across_query_microbatches: true

# ----- reconstruction execution -----
reconstruction_execution_mode: "cached_streamed"
reconstruction_query_chunk_size: 8192
reconstruction_cache_level: "static_features"
```

Keep legacy/reference comments immediately above or below the candidate options.

For Stage 4–5 options, `legacy_full` / no microbatch must remain available for numerical comparison until Stage 6 is approved.

---

# Limited validation run after all revisions

Do **not** launch a long formal run.

Prepare a limited run package under:

`_CheckNotes/Stage1_5_limited_run/`

including configs, launcher, analyzer, and README.

The run should validate execution and scaling, not establish final scientific accuracy.

---

## Limited Run A — control

Purpose: make sure the revised execution stack reproduces current behavior.

Suggested:

```text
epochs: 8–10
batch size: current stable value if memory allows
n_query_points: 4096
train_query_microbatch_size: null
optimized data path
field_normalization_mode: selected_after_full_read
reconstruction_execution_mode: cached_streamed
validation: epoch 1 and final
reconstruction: only final epoch, one fixed snapshot
NFE: 1 and 2
```

Use a fixed validation manifest.

Record:

- train loss;
- fixed-manifest validation loss;
- epoch time;
- peak memory;
- reconstruction equivalence metrics.

---

## Limited Run B — large-effective-query smoke

Purpose: verify Stage 5 actually solves the supervision/memory tradeoff.

Suggested:

```text
epochs: 5–8
same model/dataset/optimizer
n_query_points: 16384
train_query_microbatch_size: 4096
same sparse-observation settings
same optimized data path
same fixed validation manifest
```

If this is comfortably stable, optionally test `n_query_points=65536` for only 1–2 training epochs or a short benchmark rather than a full limited run.

Do not turn this into a model-quality hyperparameter study.

The main questions are:

- Does it run stably?
- Is peak memory close to the 4,096-query execution scale rather than 16,384-query monolithic scale?
- Are gradients/losses numerically correct?
- Does larger effective supervision avoid obvious convergence failure?
- What is the wall-time cost per additional supervised query?

---

## Limited reconstruction stress

Separately run an inference-only stress test:

```text
batch size: 1
query points: 1M synthetic or real 3-D points
NFE: 2
Euler
cached_streamed
```

No plotting of one million points is required.

Report:

- completion success;
- wall time;
- peak memory;
- cache memory;
- seconds per million points per NFE.

If legacy full reconstruction fits at a smaller size, compare there.

Do not require legacy full reconstruction to fit at 1M.

---

# Final deliverables before Stage 6

Codex should produce:

1. Stage-by-stage implementation summary.
2. Files changed.
3. New config keys and allowed values.
4. Explicit list of legacy/reference paths still available.
5. Fixed-manifest evaluation results.
6. Reversed-order short timing result.
7. Stage-2 data-path benchmark.
8. Stage-3 scaling CSV/JSON and interpretation.
9. Stage-4 reconstruction equivalence tests.
10. Stage-4 large-field reconstruction timing/memory.
11. Stage-5 monolithic-vs-microbatch loss/gradient/update equivalence.
12. Stage-5 memory scaling versus effective `n_query_points`.
13. Limited Run A/B results.
14. Remaining bottlenecks.
15. Exact recommendation for whether the branch is ready to proceed to Stage 6.

Also explicitly confirm:

- no GL-RBF architecture was changed;
- no RF objective was changed;
- no checkpoint parameters were reinterpreted;
- no Stage-6 architecture experiment was started.

---

# Final readiness criterion for Stage 6

Proceed to Stage 6 only if all of the following are true:

- optimized data path is validated;
- matched validation shows no material data-path correctness regression;
- reconstruction cached-streamed path is numerically equivalent;
- one-million-point reconstruction is memory-safe under streaming;
- query microbatching is gradient-equivalent;
- larger effective query supervision is practical;
- profiling shows the remaining dominant cost is genuinely model architecture / per-query computation rather than avoidable execution overhead.

At that point, Stage-6 architecture changes can be evaluated fairly against a well-optimized execution baseline.

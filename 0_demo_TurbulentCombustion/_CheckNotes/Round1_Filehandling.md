# PointCloud FFM Data-Path Efficiency Refactor with Explicit Legacy/Optimized A/B Paths

Work in repository:

`cosmos2w/PhyCoFlow_demo`

and **work specifically on branch**:

`perf/pointcloud-ffm-field-reconstruction`

Do not switch to or develop against the default branch.

The main files of interest are:

* `0_demo_TurbulentCombustion/src/train_pointcloud_ffm.py`
* `0_demo_TurbulentCombustion/src/helpers.py`
* `0_demo_TurbulentCombustion/Save_config/config_pointcloud_ffm.yaml`

You may add a small dedicated benchmark/smoke-test script under `0_demo_TurbulentCombustion/src/` if useful.
Run any test on GPU 2. DO not launch long and time consuming formal runs.

## Objective

Optimize the **data path and training input pipeline only** for `pointcloud_ffm`, especially for future 3-D cases with millions of mesh points.

Do **not** change the PointCloud FFM model architecture, GL-RBF architecture, gather mathematics, latent architecture, loss definition, Rectified Flow formulation, prior definition, or reconstruction algorithm in this task.

This is an experimental performance branch. We want unusually strong diagnostics before deciding which implementation should become permanent.

Therefore:

> Every important existing/legacy data-path behavior must remain available as an explicit configuration option.

Do not silently replace the old implementation.

The code should make the legacy and optimized paths visually obvious so that, after benchmarking, we can delete the legacy implementation cleanly.

---

# 1. Current problems to address

The current training path roughly does this:

1. `Dataset.__getitem__()` reads a complete `[N_full, C]` snapshot.
2. It normalizes the entire snapshot.
3. It clones fixed coordinates for every sample.
4. `collate_snapshots()` stacks full coordinates and full fields.
5. The full tensors are copied to GPU:

   * `coords_full = batch["coords"].to(device)`
   * `fields_full = batch["fields"].to(device)`
6. Sparse observation indices are sampled from the full mesh.
7. Query indices are sampled from the full mesh.
8. Only then are the query tensors reduced to `n_query_points`.

As a result, reducing `n_query_points` reduces model-side computation but does **not** proportionally reduce:

* HDF5 I/O,
* CPU normalization,
* coordinate duplication,
* collation,
* host memory,
* PCIe transfer,
* full-mesh index sampling.

This will become severe when `N_full` grows from ~40k points to millions of points.

There are also several smaller but important inefficiencies:

### A. Fixed coordinates are cloned for every dataset item

`TurbulentCombustionH5Dataset.__getitem__()` currently returns cloned full coordinates even though the mesh is fixed across time snapshots.

### B. Query sampling uses full `randperm`

Uniform query sampling currently uses behavior equivalent to:

```python
torch.randperm(n_pts)[:n_query]
```

This allocates/work scales with `N_full`, even if only a small fraction is selected.

### C. Observation sampling also uses full `randperm`

`build_sparse_condition()` performs another full permutation for every sample and conditioned field.

### D. Observation count sampling causes GPU synchronization

Current code contains behavior equivalent to:

```python
m = int(torch.randint(..., device=device).item())
```

The `.item()` forces a CPU/GPU synchronization inside the per-sample loop.

### E. Full tensors are transferred to GPU before query reduction

This defeats much of the purpose of `n_query_points`.

### F. DataLoader settings are not optimized for long HDF5 training

Pinned memory is used, but current code does not fully exploit:

* persistent workers,
* prefetch configuration,
* non-blocking host-to-device transfer.

### G. `obs_mix` is intrinsically expensive in the current implementation

The existing `obs_mix` path computes distances from the full mesh to observations before selecting query points.

For million-point meshes this can become an `O(N_full × M)` sampling operation.

Do not silently alter the semantics of legacy `obs_mix`.

---

# 2. Critical compatibility requirement

This task is **not** a cleanup pass.

We explicitly want legacy and optimized paths available side by side.

Where substantial old and new implementations coexist, mark them clearly using comments like:

```python
# =====================================================================
# LEGACY DATA PATH — BEGIN
# Temporary A/B reference implementation.
# Remove after optimized path is validated.
# =====================================================================

...

# =====================================================================
# LEGACY DATA PATH — END
# =====================================================================
```

and:

```python
# =====================================================================
# OPTIMIZED DATA PATH — BEGIN
# Candidate production implementation.
# =====================================================================

...

# =====================================================================
# OPTIMIZED DATA PATH — END
# =====================================================================
```

Prefer separate, clearly named functions such as:

```python
collate_snapshots_legacy(...)
collate_snapshots_optimized(...)

build_sparse_condition_legacy(...)
build_sparse_condition_optimized(...)

sample_query_indices_legacy(...)
sample_query_indices_scalable(...)
```

rather than deeply interleaving legacy and optimized logic inside many small `if` statements.

The goal is that later we can delete the complete legacy blocks with minimal risk.

Do not “refactor” the old code so heavily that we lose the ability to compare against the actual previous behavior.

---

# 3. Add an explicit data-path configuration layer

Extend `config_pointcloud_ffm.yaml` and argument parsing with an explicit experimental data-path section.

Keep the YAML flat because the existing argument/config parser is flat.

At minimum introduce something equivalent to:

```yaml
# ==========================================================
# Data-path performance / compatibility experiment
# ==========================================================

# Main profile.
# "legacy" should reproduce the historical training data path.
# "optimized" should activate the new candidate path.
data_path_mode: "optimized"       # ["legacy", "optimized"]

# Detailed overrides for component-wise ablations.
# null means "use the default associated with data_path_mode".

coord_batch_mode: null
# choices:
#   "legacy_clone"  = clone/store full coordinates in every dataset item
#   "shared_mesh"   = keep one shared fixed-mesh coordinate tensor

index_sampling_mode: null
# choices:
#   "legacy_randperm" = historical torch.randperm(N) implementation
#   "scalable"        = new sampler whose work/memory is primarily governed
#                       by number of selected points, not N_full

sampling_device: null
# choices:
#   "legacy_gpu"
#   "cpu"

field_read_mode: null
# choices:
#   "legacy_full_snapshot"
#   "indexed_union"
#
# indexed_union is experimental and must remain optional because HDF5 fancy
# indexing may or may not be faster depending on chunk layout.

gpu_transfer_mode: null
# choices:
#   "legacy_full"
#   "selected_only"

dataloader_persistent_workers: true
dataloader_prefetch_factor: 2
non_blocking_transfer: true

# Diagnostics
data_path_diagnostics: true
data_path_diag_every_n_steps: 50
data_path_diag_warmup_steps: 5
data_path_diag_max_steps_per_epoch: 10

# Avoid synchronizing CPU/GPU for tqdm every iteration.
training_log_every_n_steps: 20
```

Exact naming can be adjusted if needed, but keep the same semantics.

Implement a small resolver so that:

### `data_path_mode: legacy`

resolves to approximately:

```text
coord_batch_mode      = legacy_clone
index_sampling_mode   = legacy_randperm
sampling_device       = legacy_gpu
field_read_mode       = legacy_full_snapshot
gpu_transfer_mode     = legacy_full
persistent_workers    = historical behavior unless explicitly overridden
non_blocking_transfer = historical behavior unless explicitly overridden
```

### `data_path_mode: optimized`

resolves to approximately:

```text
coord_batch_mode      = shared_mesh
index_sampling_mode   = scalable
sampling_device       = cpu
gpu_transfer_mode     = selected_only
```

For `field_read_mode`, do **not** assume that indexed HDF5 reads are automatically better.

It is acceptable for the first optimized default to remain:

```text
field_read_mode = legacy_full_snapshot
```

while still eliminating full GPU transfer.

Then benchmark `indexed_union` independently.

Individual explicit YAML values must override profile defaults. This lets us run hybrid ablations such as:

```text
everything optimized except HDF5 read mode
```

or:

```text
legacy data path but scalable index sampling
```

---

# 4. First optimization: stop duplicating fixed mesh coordinates

The current turbulent-combustion dataset uses the same spatial mesh for all time snapshots.

Add a `shared_mesh` data path in which:

* normalized coordinates are stored once;
* raw coordinates are stored once;
* dataset samples do not clone the full mesh;
* collate does not stack `B` copies of identical coordinates;
* the trainer keeps or transfers one coordinate tensor and indexes it as required.

Do not remove `legacy_clone`.

The optimized implementation should verify that it is operating on a fixed-mesh dataset.

If the dataset abstraction could later support varying meshes, make this an explicit capability/assumption rather than silently applying shared coordinates to incompatible data.

For example, expose something conceptually like:

```python
dataset.fixed_mesh = True
```

or another simple explicit mechanism.

Do not over-engineer a generic mesh abstraction in this task.

---

# 5. Move query and observation index generation before full GPU transfer

For the optimized path, query and sparse-observation indices should be known before the complete field tensor is copied to GPU.

The intended pipeline is conceptually:

```text
time/sample index
       ↓
CPU query-index sampling
       ↓
CPU observation-index sampling
       ↓
obtain only required field values
       ↓
normalize required values
       ↓
pinned host tensors
       ↓
non-blocking GPU transfer
       ↓
model training_loss
```

The optimized path should produce directly:

```text
coords_q
fields_q

obs_coords
obs_values
obs_mask
obs_indices
obs_field_ids
```

without requiring a full `[B, N_full, C]` GPU field tensor.

The legacy path must continue to work exactly as before.

---

# 6. Implement a scalable unique-index sampler

Keep the historical full `randperm` implementation as:

```text
index_sampling_mode = legacy_randperm
```

Add a second sampler for the common regime:

```text
K << N_full
```

Its memory and runtime should be governed mainly by `K`, rather than always allocating an `N_full` permutation.

Requirements:

* sample without replacement;
* reproducible from the configured seed;
* support query selection and observation selection;
* work efficiently for `N_full ~ 1e6` and `K ~ 10^2–10^5`;
* return sorted indices when sorted HDF5 indexing is useful;
* avoid hidden `O(N_full)` temporary arrays if practical.

Choose an appropriate implementation after benchmarking.

Possible strategies include a Floyd-style sampler, oversampled random integers plus deduplication for sparse selections, or another efficient tested method.

Do not replace the old sampler.

Also avoid device `.item()` calls inside per-sample loops in the optimized path.

Sampling observation counts and indices should preferably occur entirely on CPU in the optimized path.

---

# 7. Separate sampling of indices from construction of tensors

Currently `build_sparse_condition()` combines:

* random observation-count generation,
* random index generation,
* coordinate gather,
* value gather,
* padding/masking.

Refactor the optimized path so that index sampling can happen independently.

For example, conceptually:

```python
obs_layout = sample_sparse_observation_indices(...)
query_indices = sample_query_indices(...)

batch = materialize_selected_batch(
    time_indices,
    query_indices,
    obs_layout,
)
```

This makes it possible to:

* sample before GPU transfer;
* use selected HDF5 reads;
* reuse fixed observation layouts;
* reproduce exactly the same layout between legacy and optimized diagnostic runs.

Keep the legacy `build_sparse_condition()` path available.

---

# 8. Add an optional indexed HDF5 read path

Implement an experimental:

```text
field_read_mode = indexed_union
```

For each snapshot, form the union of:

```text
query indices
+
observation indices
```

and read only those field values where practical.

Important:

HDF5 fancy indexing performance depends strongly on dataset chunking and access pattern.

Therefore:

* keep `legacy_full_snapshot`;
* do not claim `indexed_union` is always faster;
* instrument both modes;
* inspect and log the relevant HDF5 dataset shape/chunk layout at startup.

For indexed reads:

1. deduplicate selected point indices;
2. sort them if h5py requires/benefits from sorted indexing;
3. perform one union read rather than many tiny reads;
4. map the returned values back to query and observation layouts;
5. avoid reading query and observation data separately when they overlap.

If the current HDF5 layout makes indexed access pathologically slow, preserve the implementation for diagnostic comparison but do not force it as the optimized default.

A useful intermediate optimized path is:

```text
read full snapshot to CPU
→ select queries/sensors on CPU
→ transfer selected tensors only
```

This already removes the large GPU-transfer penalty.

---

# 9. Preserve `obs_mix` semantics

Do not silently rewrite the historical `obs_mix` algorithm.

The current `obs_mix` mode computes full-mesh distance information and therefore may remain expensive.

For this task:

* keep the existing `obs_mix` implementation explicitly available as legacy behavior;
* continue supporting `uniform`;
* optimized uniform sampling must not fall back to full-mesh `cdist`.

If you can implement a clean scalable spatial-index version of `obs_mix` without substantial unrelated complexity, add it under a **new explicit name**, for example:

```text
query_sampling: "obs_mix_indexed"
```

but do not replace `obs_mix`.

This optional addition is lower priority than the main uniform data-path optimization.

---

# 10. Improve DataLoader plumbing without changing training semantics

For the optimized path, support:

```python
persistent_workers=True
```

when:

```text
num_workers > 0
```

Expose `prefetch_factor`.

Continue using pinned memory.

Use:

```python
tensor.to(device, non_blocking=True)
```

for tensors coming from pinned host memory when:

```text
non_blocking_transfer = true
```

Be careful with HDF5 worker handles.

Each worker must maintain its own safe HDF5 handle; do not share a live `h5py.File` object across forked workers in an unsafe manner.

Preserve lazy worker-local opening behavior or improve it if necessary.

If persistent workers require resetting/reopening handles, handle that safely.

---

# 11. Remove unnecessary synchronization in the optimized training loop

The current training loop converts the loss to CPU every batch and updates tqdm every batch.

For the optimized path, only perform synchronized scalar logging every:

```text
training_log_every_n_steps
```

Do not change the actual accumulated epoch loss.

It is fine to maintain a GPU-side or unsynchronized accumulator and materialize periodically.

Similarly, avoid unnecessary `.item()` / `.cpu()` calls in inner data-path loops.

Keep the legacy logging behavior available if needed for exact historical profiling, but this does not require a separate large code path if a simple config switch is sufficient.

Do not introduce mixed precision in this task.

---

# 12. Add comprehensive data-path diagnostics

This branch is explicitly for performance diagnosis.

Add lightweight instrumentation that can be enabled/disabled from YAML.

Measure at least the following phases where applicable:

```text
loader_wait
index_sampling
HDF5_read
CPU_normalization
CPU_materialization / collate
host_to_device
sparse_condition_materialization
query_materialization
pre_model_total
model_forward
backward
optimizer_step
total_training_step
```

The model timings are included only to understand the fraction of total step time consumed by the data path. Do not alter the model.

For CUDA memory, record at least:

```text
peak_allocated_MB
peak_reserved_MB
```

Where useful, also record:

```text
allocated_before_model_MB
allocated_after_materialization_MB
```

Use appropriate timing mechanisms:

* `time.perf_counter()` for CPU stages;
* CUDA events or explicit synchronization only on diagnostic steps for GPU timing.

Do **not** synchronize every training iteration just to collect diagnostics.

Diagnostics should sample only configured steps.

Save results under the run directory, for example:

```text
data_path_diagnostics.csv
data_path_diagnostics.json
```

Useful identifying columns include:

```text
epoch
step
data_path_mode
coord_batch_mode
index_sampling_mode
sampling_device
field_read_mode
gpu_transfer_mode

batch_size
N_full
N_query
N_obs_total

loader_wait_ms
index_sampling_ms
hdf5_read_ms
cpu_normalization_ms
h2d_ms
pre_model_total_ms
model_forward_ms
backward_ms
optimizer_ms
total_step_ms

gpu_peak_allocated_mb
gpu_peak_reserved_mb
```

Also print a concise epoch-level summary such as:

```text
[data-path]
mode=optimized
Nfull=1,000,000
Nq=50,000
pre-model=...
forward=...
backward=...
peak=...
```

Do not flood stdout with per-step details.

---

# 13. Add a dedicated A/B benchmark utility

Add a small utility such as:

`src/benchmark_pointcloud_data_path.py`

The purpose is to benchmark the data pipeline independently from a long training run.

It should use the real dataset and config definitions where possible.

Support comparing at minimum:

```text
legacy
optimized
```

and preferably hybrid component overrides.

Allow configurable sweeps of:

```text
batch size
n_query_points
observation count
number of benchmark iterations
```

Suggested default query sweep:

```text
4096
16384
65536
```

and full-field if the dataset is small enough.

Do not hard-code assumptions that only apply to the current ~40k 2-D dataset.

The benchmark should report:

```text
samples/sec
selected query points/sec
pre-model latency
HDF5 latency
H2D latency
peak GPU memory
```

and save a CSV.

If easy to implement, report ratios such as:

```text
optimized / legacy step time
optimized / legacy pre-model time
optimized / legacy GPU memory
```

No plotting framework is required unless already convenient.

---

# 14. Correctness / equivalence checks

Because sampling is stochastic, do not simply run two random batches and compare losses.

Create a diagnostic pathway where the same:

```text
time indices
query indices
observation indices
observation counts
```

are fed through the legacy and optimized tensor-materialization paths.

Then verify before the model call that corresponding tensors agree:

```text
coords_q
fields_q
obs_coords
obs_values
obs_mask
obs_indices
obs_field_ids
```

Use exact equality where appropriate and tight `allclose` for floating values.

For a stronger optional smoke test:

1. load one model state;
2. reset the PyTorch RNG to the same state before each `training_loss()` call;
3. pass identical selected tensors from both paths;
4. verify the resulting loss agrees within numerical tolerance.

This is particularly important because the RFF prior samples random coefficients.

The optimized data path should not change model inputs given the same selected indices.

---

# 15. Important behavior of `n_query_points`

After this refactor, verify and document the effective scaling.

With:

```text
gpu_transfer_mode = selected_only
```

GPU memory and PCIe traffic before model evaluation should primarily scale with:

```text
N_query + N_obs
```

rather than:

```text
N_full
```

With:

```text
field_read_mode = indexed_union
```

CPU I/O should also become more dependent on selected points, subject to HDF5 chunking.

The benchmark should make this distinction visible.

---

# 16. Keep reconstruction behavior unchanged in this round

Do not implement model-side reconstruction caching, end-to-end query streaming, latent caching, or GL-RBF architecture changes in this task.

The only reconstruction-related change allowed is avoiding obviously unnecessary dataset coordinate clones or data loading duplication if it follows naturally from the shared-mesh dataset change.

The full reconstruction model path itself must remain unchanged.

We will optimize that separately after the training data path has been measured.

---

# 17. Do not include these changes in this task

Do not change:

* `GL_rbf_ENH`
* `topk_rbf`
* `topk_rbf_glres`
* `gather_topk`
* KeOps logic
* query-to-latent readout
* latent count
* hidden dimensions
* Fourier bands
* fusion head
* prior mathematics
* RF training target
* loss function
* ODE integration
* mixed precision
* DDP/multi-GPU
* gradient accumulation/query microbatching

Those are separate later experiments.

This round is specifically intended to isolate **data-path engineering**.

---

# 18. Code quality requirements

Do not turn `train_pointcloud_ffm.py` into one large collection of conditionals.

Prefer well-named helpers and explicit boundaries.

Legacy code should be visibly marked as temporary.

Optimized code should also be visibly marked.

Where possible, resolve the selected modes once at startup rather than repeatedly parsing strings inside hot loops.

At startup, print the resolved data-path configuration clearly, e.g.:

```text
[*] PointCloudFFM data path:
    profile              = optimized
    coord_batch_mode     = shared_mesh
    index_sampling_mode  = scalable
    sampling_device      = cpu
    field_read_mode      = legacy_full_snapshot
    gpu_transfer_mode    = selected_only
    persistent_workers   = true
    non_blocking_transfer= true
```

This is essential so saved logs are interpretable later.

Also save the resolved values in `args.json` / run configuration exactly as the trainer currently saves other arguments.

---

# 19. Config template must explicitly show both recommended profiles

In `Save_config/config_pointcloud_ffm.yaml`, include a clearly marked commented reference section such as:

```yaml
# ==========================================================
# LEGACY DATA PATH REFERENCE
# Temporary for controlled A/B comparison.
# ==========================================================
#
# data_path_mode: "legacy"
# coord_batch_mode: "legacy_clone"
# index_sampling_mode: "legacy_randperm"
# sampling_device: "legacy_gpu"
# field_read_mode: "legacy_full_snapshot"
# gpu_transfer_mode: "legacy_full"

# ==========================================================
# OPTIMIZED DATA PATH CANDIDATE
# Keep this while benchmarking against legacy.
# ==========================================================

data_path_mode: "optimized"
coord_batch_mode: "shared_mesh"
index_sampling_mode: "scalable"
sampling_device: "cpu"

# Initially benchmark both of these:
field_read_mode: "legacy_full_snapshot"
# field_read_mode: "indexed_union"

gpu_transfer_mode: "selected_only"

dataloader_persistent_workers: true
dataloader_prefetch_factor: 2
non_blocking_transfer: true

data_path_diagnostics: true
data_path_diag_every_n_steps: 50
data_path_diag_warmup_steps: 5
training_log_every_n_steps: 20
```

Do not delete the old settings after adding the new ones.

The intent of this branch is that we can later run a matrix such as:

```text
A. fully legacy
B. legacy + shared coordinates only
C. B + scalable CPU sampling
D. C + selected-only GPU transfer
E. D + persistent workers / nonblocking transfer
F. E + indexed HDF5 reads
```

and identify where the actual speedups originate.

---

# 20. Deliverables

After implementation, report:

1. Files changed.
2. Exact new YAML options and allowed values.
3. Which original behaviors remain available and how to activate them.
4. Any behavior that could not be preserved exactly and why.
5. How the optimized data flow differs from legacy.
6. HDF5 chunk/layout information discovered from the current dataset.
7. Results from a short legacy-vs-optimized benchmark on the available dataset.
8. Timing breakdown of the major data stages.
9. Peak GPU-memory comparison.
10. Any cases where `indexed_union` is slower than full-snapshot HDF5 reads.
11. Correctness/equivalence test results using identical query/observation indices.
12. Remaining bottlenecks after the data-path improvements.

Finally, show the relevant `git diff` summary and explicitly confirm that no GL-RBF/model architecture code or RF objective was modified.

The main purpose of this work is **measurement and clean A/B comparability**, not merely producing the shortest final implementation.

---

# Implementation progress

This section is maintained as each implementation/validation gate passes.

## Gate 1 — Specification and current-path audit: PASSED

Completed on 2026-08-20.

Evidence:

* Confirmed work is on `perf/pointcloud-ffm-field-reconstruction`.
* Read this complete specification before editing implementation files.
* Located the historical fixed-coordinate clone and full-snapshot normalization in
  `src/helpers.py::TurbulentCombustionH5Dataset.__getitem__`.
* Located the historical per-sample GPU `randint(...).item()` and full
  `randperm(N_full)` observation sampler in `src/helpers.py::build_sparse_condition`.
* Located full-coordinate/full-field collation, full host-to-device transfer,
  post-transfer query sampling, and per-step synchronized loss logging in
  `src/train_pointcloud_ffm.py`.
* Located all three DataLoader construction sites, including the separate
  direct-coherence epoch loader.
* Inspected the active HDF5 layout. `fields` has shape
  `(1, 10000, 40300, 1, 1, 5)`, dtype `float32`, contiguous layout
  (`chunks=None`), and no compression. `coordinates` has shape
  `(40300, 1, 1, 3)`, also contiguous and uncompressed.
* Chosen implementation boundary: retain visibly marked legacy functions and put
  the candidate sampling/materialization/diagnostic code in a dedicated
  `src/pointcloud_data_path.py` module, with thin integration in the trainer and
  explicit dataset modes in `helpers.py`.

Next gate: implement and unit-validate configuration resolution, scalable unique
sampling, independent sparse-layout sampling, shared-mesh collation, and selected
batch materialization while retaining the legacy path.

## Gate 2 — Reusable data-path components: PASSED

Completed on 2026-08-20.

Implemented:

* Added flat profile resolution with explicit component overrides and validation.
* Preserved full `randperm` sampling as `sample_unique_indices_legacy` and added
  an O(K)-scale sparse-regime sampler as `sample_unique_indices_scalable`.
* Separated CPU observation-count/index layout sampling from tensor gathering.
* Added fixed-mesh capability declaration, shared normalized/raw coordinates,
  and worker-PID-aware lazy HDF5 handle reopening.
* Preserved cloned-coordinate dataset items as `coord_batch_mode=legacy_clone`.
* Added one-union, sorted/deduplicated HDF5 reads with mapping back to query and
  observation layouts, while keeping full-snapshot reads as the optimized default.
* Added selected tensor materialization and hybrid full-transfer materialization
  helpers under clearly marked optimized/legacy boundaries.

Validation evidence:

* `python -m py_compile` passed for `src/pointcloud_data_path.py`,
  `src/helpers.py`, and `src/train_pointcloud_ffm.py`.
* `pytest -q tests/test_pointcloud_data_path.py`: **6 passed**.
* The scalable sampler was directly exercised at `N_full=1,000,000`,
  `K=100,000`; it returned exactly 100,000 sorted unique in-range indices and
  reproduced bit-for-bit with the same seeded CPU generator.
* Identical query/observation layouts produced matching `coords_q`, `fields_q`,
  `obs_coords`, `obs_values`, `obs_mask`, `obs_indices`, and `obs_field_ids`
  through both `legacy_full_snapshot` and `indexed_union` selected materializers.

Next gate: validate DataLoader integration, selected-only/non-blocking transfer,
sampled diagnostics, saved resolved configuration, legacy launch compatibility,
and the standard training loop without changing the model/RF objective.

## Gate 3 — Trainer, DataLoader, transfer, and diagnostics integration: PASSED

Completed on 2026-08-20.

Implemented:

* Added explicit legacy and optimized collators plus hybrid component support.
* Added persistent-worker and prefetch plumbing only when `num_workers > 0`;
  pinned-memory behavior remains enabled on CUDA systems.
* Added selected-only `.to(..., non_blocking=...)` transfers and shared-mesh
  expansion without stacking full coordinate copies.
* Added full-transfer hybrid support so CPU/scalable layouts can be benchmarked
  before enabling selected-only transfer.
* Retained the exact GPU observation/query sampling path for the full legacy
  profile, including historical `obs_mix` semantics.
* Changed optimized standard-training loss logging to synchronize only at the
  configured interval, while accumulating the exact epoch loss on-device and
  materializing it once at epoch end. Legacy defaults to every-step logging.
* Added sampled CPU/CUDA phase timing, peak allocated/reserved memory, per-epoch
  summaries, and `data_path_diagnostics.csv`/`.json` persistence.
* Resolved settings are printed at startup and copied into `args` before
  `args.json` is written.
* The active YAML has no unrecognized keys and resolves to the intended optimized
  candidate (`shared_mesh`, scalable CPU sampling, full CPU snapshot read,
  selected-only GPU transfer).

Validation evidence:

* Physical GPU inventory confirmed GPU 2 is an NVIDIA RTX 6000 Ada Generation.
* With `CUDA_VISIBLE_DEVICES=2`, the combined focused/regression suite reported
  **16 passed**. This includes explicit CUDA runs of both legacy and optimized
  trainer loops, selected-only transfers, backward/optimizer steps, and nonzero
  CUDA peak-memory diagnostics.
* The same trainer smoke tests also passed on CPU.
* Both legacy and optimized diagnostic runs produced CSV and JSON files with all
  requested major stage/memory columns.
* Existing CONFIG gradient-update regression tests remained green.

Compatibility note:

* Standard training supports the complete optimized path. Direct-coherence
  training still requires `gpu_transfer_mode=legacy_full` because its auxiliary
  rollout samples additional reference points after the main data batch is
  formed; shared coordinates and CPU/scalable index sampling remain usable there.
  This is an explicit launch-time error rather than a silent behavior change.

Next gate: run the real-dataset A/B benchmark on GPU 2 across query sizes and
both HDF5 read modes, and retain a durable CSV with ratios and phase breakdowns.

## Gate 4 — Real-dataset GPU 2 A/B benchmark: PASSED

Completed on 2026-08-20.

Benchmark setup:

* Physical GPU 2 (NVIDIA RTX 6000 Ada Generation), exposed as logical `cuda:0`
  with `CUDA_VISIBLE_DEVICES=2`.
* Real `Dataset/Merged_COTU0U1P.h5`, batch size 4, 256 observations per sample,
  two DataLoader workers, one warmup and three measured iterations.
* Query sizes: 4,096; 16,384; and full mesh (40,300).
* Durable results: `_CheckNotes/Round1_data_path_benchmark_gpu2.csv`.
* The full-mesh `indexed_union` run was skipped by default because its sorted
  union is the complete mesh, making fancy indexing both redundant and already
  demonstrably pathological. The utility exposes `--benchmark-indexed-full` if
  this deliberately expensive case is ever needed.

Key measured results (milliseconds and MiB):

| Path | N query | total step | HDF5 stage work | H2D | peak allocated | step / legacy | memory / legacy |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 4,096 | 14.62 | 2.67 | 0.34 | 6.92 | 1.000 | 1.000 |
| optimized, full read | 4,096 | 2.80 | 2.30 | 0.65 | 0.54 | 0.191 | 0.077 |
| optimized, indexed union | 4,096 | 336.54 | 720.15 | 1.51 | 0.54 | 23.02 | 0.077 |
| legacy | 16,384 | 15.16 | 2.60 | 0.40 | 7.46 | 1.000 | 1.000 |
| optimized, full read | 16,384 | 1.99 | 1.81 | 1.83 | 2.04 | 0.132 | 0.273 |
| optimized, indexed union | 16,384 | 3,855.02 | 10,982.70 | 0.95 | 2.04 | 254.36 | 0.273 |
| legacy | 40,300 | 13.41 | 2.26 | 0.33 | 11.10 | 1.000 | 1.000 |
| optimized, full read | 40,300 | 4.67 | 1.78 | 0.96 | 4.96 | 0.349 | 0.446 |

Interpretation:

* At 4,096 queries, optimized/full-read is about **5.2x faster** in measured
  pre-model step wall time and uses about **92.3% less peak allocated GPU memory**.
* At 16,384 queries, it is about **7.6x faster** and uses about **72.7% less peak
  allocated GPU memory**.
* Even at a full-field query it is about **2.9x faster** in this pipeline-only
  benchmark, largely from shared coordinates/CPU layout and avoiding duplicate
  legacy GPU materialization, though selected-query scaling naturally disappears.
* `indexed_union` is decisively slower (23.0x at 4,096 and 254.4x at 16,384
  queries in observed step time) on the current contiguous dataset and must
  not be the optimized default. It preserves a useful diagnostic implementation
  for future chunked million-point datasets.
* With worker prefetch, per-batch HDF5/CPU stage work is measured inside workers
  and can overlap the consumer wall time; therefore stage-work columns need not
  sum to `total_step_ms`. The latter is the observed consumer-side latency.

Benchmark validation also caught and fixed a full-query pin-memory boundary bug:
an expanded zero-stride all-point index view is now materialized as contiguous
per-batch storage. A dedicated regression test covers this case; the GPU 2 focused
suite passed afterward.

The final benchmark rerun explicitly released each previous batch before resetting
CUDA peak counters, so the table does not include stale-tensor overlap from the
preceding iteration.

Next gate: perform the final requirement-by-requirement audit, run the complete
available test suite and diff/scope checks, and document any remaining limitations
or bottlenecks before declaring this round complete.

## Gate 5 — Final scope, correctness, and deliverable audit: PASSED

Completed on 2026-08-20.

Final validation:

* Complete available suite on physical GPU 2: **17 passed in 7.37 s**.
* `py_compile` passed for both new utilities and the modified helper/trainer.
* Ruff passed with no findings for the two new modules and dedicated test file.
* `git diff --check` passed.
* Active YAML parsing found zero unrecognized keys and the resolved candidate
  settings match the printed/saved profile.
* Identical-index equivalence tests cover both selected materializers and all
  seven pre-model tensors. Exact equality is used for indices/masks/coordinates;
  normalized floating values use `rtol=atol=1e-6`.
* Optimized full-snapshot reconstruction access is explicitly tested, preserving
  the existing visualization/reconstruction contract without restoring per-item
  coordinate clones.

### Files changed for this round

* `Save_config/config_pointcloud_ffm.yaml`
* `src/helpers.py`
* `src/train_pointcloud_ffm.py`
* `src/pointcloud_data_path.py` (new)
* `src/benchmark_pointcloud_data_path.py` (new, executable)
* `tests/test_pointcloud_data_path.py` (new)
* `_CheckNotes/Round1_Filehandling.md` (this gate log)
* `_CheckNotes/Round1_data_path_benchmark_gpu2.csv` (benchmark evidence)

The pre-existing changes in `Proj_MultiFieldReconstruction/ModelExplain.md` and
`Proj_MultiFieldReconstruction/README.md` were not touched by this round.

### Exact configuration surface

* `data_path_mode`: `legacy` or `optimized`.
* `coord_batch_mode`: `legacy_clone` or `shared_mesh`.
* `index_sampling_mode`: `legacy_randperm` or `scalable`.
* `sampling_device`: `legacy_gpu` or `cpu`.
* `field_read_mode`: `legacy_full_snapshot` or `indexed_union`.
* `gpu_transfer_mode`: `legacy_full` or `selected_only`.
* `dataloader_persistent_workers`: boolean or `null` for profile default.
* `dataloader_prefetch_factor`: positive integer or `null`.
* `non_blocking_transfer`: boolean or `null` for profile default.
* `data_path_diagnostics`: boolean.
* `data_path_diag_every_n_steps`: positive integer.
* `data_path_diag_warmup_steps`: nonnegative integer.
* `data_path_diag_max_steps_per_epoch`: nonnegative integer.
* `training_log_every_n_steps`: positive integer or `null` for profile default.

Incompatible combinations fail explicitly: selected-only transfer needs CPU
sampling; scalable sampling needs CPU sampling; indexed-union reads need CPU
sampling plus selected-only transfer.

### Preserved behaviors and activation

The complete historical path remains available with:

```yaml
data_path_mode: "legacy"
coord_batch_mode: "legacy_clone"
index_sampling_mode: "legacy_randperm"
sampling_device: "legacy_gpu"
field_read_mode: "legacy_full_snapshot"
gpu_transfer_mode: "legacy_full"
dataloader_persistent_workers: false
dataloader_prefetch_factor: null
non_blocking_transfer: false
training_log_every_n_steps: 1
```

This retains full snapshot reads/normalization, per-item coordinate clones,
stacked full coordinates/fields, full GPU transfer, GPU `randperm`, historical
observation-count sampling, historical `obs_mix`, and every-step loss display.
Component values may be changed individually for the A–F ablation matrix.

The only unsupported combination is selected-only transfer in direct-coherence
training. That auxiliary objective needs additional full reference fields after
main-batch selection, so it now raises a clear launch-time error and instructs the
user to select `gpu_transfer_mode: legacy_full`. Direct coherence itself and all
of its mathematics remain unchanged.

### Optimized flow

The candidate standard-training flow is now:

```text
sample/time ids
  -> CPU observation counts and unique observation/query indices
  -> one worker-local full snapshot read (default) or one sorted union read
  -> CPU normalization and selected tensor materialization
  -> pinned selected tensors only
  -> nonblocking H2D of N_query + N_obs data
  -> unchanged model.training_loss / backward / optimizer
```

Coordinates remain once on the fixed-mesh dataset. The HDF5 handle is opened
lazily per process, detects fork PID changes, and is removed during spawn
pickling. Full-snapshot reconstruction uses a separate accessor and does not
depend on the optimized training-item representation.

### Remaining bottlenecks and decisions

* `legacy_full_snapshot` still scales HDF5 I/O and CPU normalization with
  `N_full`; its advantage here is the contiguous on-disk layout and sequential
  read speed. It remains the recommended candidate default for this dataset.
* CPU unique-index sampling is visible stage work (especially at small query
  counts in one measured worker batch), though worker prefetch hid it effectively
  in consumer wall time. It is now independently measurable for future tuning.
* `indexed_union` is dominated by h5py fancy indexing on this contiguous file and
  should only be reconsidered for differently chunked future datasets.
* Historical `obs_mix` remains O(N_full x M) and is intentionally not rewritten.
* Direct-coherence training still needs full reference tensors.
* Model forward/backward, GL-RBF gather/readout, reconstruction streaming/caching,
  and RF/prior work remain deliberately outside this round.

### Diff/scope confirmation

Tracked round-specific diff summary:

```text
Save_config/config_pointcloud_ffm.yaml |  41 lines added
src/helpers.py                         | 166 lines changed
src/train_pointcloud_ffm.py            | 489 lines changed
tracked subtotal                       | 608 insertions, 88 deletions
```

New implementation/test files contain 734, 345, and 291 lines respectively;
the benchmark CSV contains eight result rows plus its header.

`src/Model.py`, `src/direct_coherence_loss.py`, and all GL-RBF, top-k gather,
KeOps, query-to-latent, prior, Rectified Flow target/loss, ODE, and reconstruction
algorithm implementations have **no diff**. This round modified only data-path,
loader, transfer, diagnostics, benchmark, compatibility-access, configuration,
test, and documentation code.

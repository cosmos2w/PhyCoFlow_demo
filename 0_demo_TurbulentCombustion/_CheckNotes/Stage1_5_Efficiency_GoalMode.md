# PointCloud FFM Stages 1–5 Efficiency Progress

Baseline: `0e5854e` on `perf/pointcloud-ffm-field-reconstruction`.

Hard scope: execution and data efficiency only. GL-RBF mathematics, top-k/KeOps
semantics, the RFF prior, Rectified-Flow target/loss, solver formulas, observation
consistency, and checkpoint parameter interpretation remain unchanged. Stage 6
architecture experiments are excluded.

## Stage status

| Stage | Status | Gate evidence |
|---|---|---|
| 1. Matched evaluation and order control | Passed | Fixed manifest, 192 paired GPU-0 evaluations, 12+12 epoch reversed-order control |
| 2. Data-path polish | Passed | Selected normalization equivalence, append diagnostics, real-data A/B benchmark |
| 3. Million-point scaling | Passed | 27 data-path rows + 9 current-model GPU rows through 1M/65,536 queries |
| 4. Cached/streamed reconstruction | Passed | Tight equivalence matrix + real checkpoint + 1M-query stress |
| 5. Query-microbatch training | Passed | Full RF bridge once, all-gradient/Adam equivalence, 65,536-query GPU steps |
| Limited validation package | Passed | 8-epoch control + 5-epoch 16k/4k run + fixed manifest + 1M stress |

## Stage 1 log

### Baseline audit

- Branch and baseline SHA match the specification exactly.
- Existing Round 1 legacy and optimized checkpoints are present locally.
- Existing final validation used independently sampled layouts/RF randomness.
- Existing timing order was legacy then optimized; the order-control launcher
  will run optimized then legacy on the same physical GPU.

### Commands and evidence

- `python -m py_compile` passed for the manifest generator, matched evaluator,
  and Stage-1 tests.
- `pytest tests/test_pointcloud_eval_manifest.py tests/test_pointcloud_data_path.py -q`:
  **12 passed in 4.52 s**.
- Real manifest: 64 validation snapshots, 4,096 queries per snapshot, 193–383
  observations, checksum
  `392806184e0257f95f8d7a550ef1fb9ca85a1bd7fa8537d5471807763f1a0822`.
- Optimized-first/legacy-second configs differ only in explicit data-path
  components, run ID, and output directory.
- The guarded GPU-0 launcher is queued because another process currently owns
  GPU 0; it will not co-locate the controlled jobs.
- A real-checkpoint CPU smoke evaluation loaded the epoch-100 GL-RBF checkpoint
  strictly and completed an RF-loss call. Evaluating the same checkpoint through
  two separately constructed models with two RF seeds produced exactly zero
  paired loss difference, confirming evaluator RNG restoration and input reuse.
- Full fixed-manifest checkpoint result (64 samples, batch 8, three RF repeats,
  24 paired batch evaluations): legacy `0.61161387 +/- 0.08198033`, optimized
  `0.60958297 +/- 0.08193937`; optimized-minus-legacy paired mean
  `-0.00203090`, paired standard deviation `0.00615242`, maximum absolute paired
  difference `0.01851249`. The optimized mean is 0.33% lower, so the earlier
  unmatched +2.05% final validation result is not a systematic material
  regression under controlled layouts and RF randomness.
- Official per-manifest result (batch size 1, 64 manifests x three RF repeats =
  192 paired evaluations): legacy `0.65306722 +/- 0.24308206`, optimized
  `0.65083629 +/- 0.24162502`; optimized-minus-legacy paired mean
  `-0.00223093`, paired standard deviation `0.02398152`, maximum absolute paired
  difference `0.15892988`. The optimized checkpoint mean remains 0.34% lower.
  Machine-readable rows and summary are in
  `_CheckNotes/Stage1_matched_checkpoint_eval.csv/.json`.
- GPU-0 confirmation used the same 192 controlled pairs: legacy mean
  `0.60540086`, optimized mean `0.60486497`, paired optimized-minus-legacy mean
  `-0.00053589` (optimized 0.09% lower), paired standard deviation `0.02337562`.
- Reversed timing order used optimized first and legacy second for 12 epochs
  each. The updated goal permitted a matched batch-size reduction to 96 so the
  jobs could share GPU 0 with a pre-existing 10.6 GiB/100%-utilization process.
  Initial co-tenant state is recorded in `Stage1_order_runtime/gpu0_initial_state.csv`;
  therefore these are relative order-control timings, not clean absolute times.
- Steady epochs 3–12: optimized `61.2904 s/epoch`, legacy `86.8460 s/epoch`.
  Optimized remained **29.43% faster (1.417x)** even when it ran first. Both
  histories contain exactly 12 epochs, both jobs exited zero, and no runtime/OOM
  errors were found.

### Stage-1 gate decision: PASSED

- Manifest regeneration is bit-identical and checksum protected.
- Both checkpoints consume the same materialized tensors.
- Per-manifest RF loss uses identical CPU/CUDA RNG resets for each pair.
- Matched evaluation shows no material optimized-checkpoint regression.
- The optimized timing advantage survives reversed order.
- No model or RF mathematics changed in Stage 1.
- Complete regression suite on physical GPU 0: **20 passed in 15.73 s**.

## Stage 2 log

### Implementation

- Added independent `field_normalization_mode` control. The optimized default
  performs one sequential full HDF5 snapshot read, forms one sorted query/observation
  union, gathers raw rows, and normalizes only that union. The exact historical
  full-snapshot normalization remains available.
- Added append-only CSV/JSONL diagnostics with a bounded current-epoch memory
  window and compact latest/cumulative summary JSON. `legacy_rewrite` remains
  available for controlled comparison.
- Moved standard-training `zero_grad(set_to_none=True)` before forward and added
  `allocated_before_model_mb` diagnostics. Direct-coherence update ordering was
  intentionally left unchanged.
- Added explicit active and legacy defaults in the main YAML and an
  `optimized_fullnorm` benchmark profile.

### Commands and evidence

- Focused suite: `pytest -q tests/test_pointcloud_data_path.py`: **13 passed in
  10.45 s** (project conda environment).
- Complete regression suite on physical GPU 0: **22 passed in 10.78 s**.
- Real contiguous HDF5 benchmark: `N_full=40,300`, batch 4, `M=256`, GPU 0,
  2 warmups + 8 measured iterations, isolated process per normalization mode.
- At the active `N_query=4,096`, selected normalization reduced normalization
  time from `2.600 ms` to `1.866 ms` (**28.2%**), pre-model latency from
  `9.559 ms` to `7.688 ms` (**19.6%**), and total measured data-path step from
  `10.544 ms` to `8.025 ms` (**23.9%**).
- Isolated maximum host RSS was `610,900 KiB` for full normalization versus
  `605,232 KiB` for selected normalization (**0.93% lower**).
- At `N_query=16,384` on this small 40,300-point mesh, union construction and
  indexing outweigh avoided normalization: total was `13.095 ms` selected vs
  `11.776 ms` full normalization. This crossover is documented rather than
  hidden; the Stage-2 optimized default targets the active 4,096-query case and
  future much larger meshes. Stage 3 measures the scaling boundary explicitly.
- Machine-readable output and the exact benchmark commands are under
  `_CheckNotes/Stage2_data_path/`.

### Stage-2 gate decision: PASSED

- Both full-read normalization paths match legacy selected tensors at
  `rtol=atol=1e-6`; indexed-union remains explicit and requires selected
  normalization.
- Active-workload CPU normalization/pre-model time and host RSS improve.
- Append diagnostics perform O(new rows) persistence and O(current epoch) summary
  work rather than rewriting an ever-growing history.
- Standard training passes and asserts gradients are cleared before forward.
- No model, checkpoint, GL-RBF, top-k, KeOps, RFF, or RF objective mathematics
  changed.

## Stage 3 log

### Implementation and evidence

- Added `src/benchmark_pointcloud_scaling.py` with separate `data` and `model`
  benchmark classes and a stable CSV/JSON schema for all required phases,
  throughput metrics, host RSS, and CUDA allocation/reservation.
- The data sweep uses the real 40,300-point HDF5 dataset. Because no formal
  250k/1M HDF5 dataset exists locally, those rows use an explicitly labeled
  in-memory expansion of one real snapshot; their host-clone time is not claimed
  as HDF5 I/O time.
- The model sweep uses the active GL_rbf_ENH, topk_rbf_glres, KeOps, dimensions,
  and query chunk setting on synthetic GPU-resident 3-D tensors. It changes no
  architecture or mathematics.
- Focused benchmark/schema tests: **15 passed in 10.50 s**.
- Complete regression suite on physical GPU 0: **24 passed in 10.66 s**.
- Data sweep: 27 rows across `N_full={40,300, 250k, 1M}`,
  `N_query={4,096, 16,384, 65,536/full}`, and `M={256,512,1024}`.
- Model sweep: 9 rows across `N_query={4,096,16,384,65,536}` and
  `M={256,512,1024}`; every row completed without OOM.
- At fixed 4,096 queries, mean pre-model time changes from `17.99 ms` at 40.3k
  full points to `24.44 ms` at 1M, while selected GPU inputs stay `0.582 MB`.
- Model step time is `61.15–64.53 ms` at 4,096 queries and
  `709.39–801.20 ms` at 65,536. Model peak allocation grows from `255–271 MB`
  to `3.02–3.04 GB`.
- At the largest measured combination, the 1M-point data path averages
  `45.73 ms`, while the 65,536-query model costs `0.71–0.80 s`; query execution
  is decisively dominant.
- Raising observations from 256 to 1,024 adds 5.5% step time at 4,096 queries
  and 12.4% at 65,536. KeOps controls pairwise memory, but exact neighbor work
  remains measurable.
- Full tables, raw CSV/JSON, analyzer, interpretation, limitations, and exact
  commands are under `_CheckNotes/Stage3_scaling/`.

### Stage-3 gate decision: PASSED

- Full-mesh and selected-query costs are independently measured through the
  requested scales.
- The crossover is unambiguous: current per-query model work dominates before
  million-point data handling becomes the main constraint.
- The near-linear query-activation memory curve directly motivates Stage 4
  streaming and Stage 5 microbatching.
- The optional HDF5 layout experiment was not performed because it would require
  large temporary storage and does not affect the measured execution bottleneck.

## Stage 4 log

### Implementation

- Added differentiable `prepare_condition_context`, cache-aware
  `prepare_query_context`, and complete `forward_query_chunk` APIs to the existing
  GL-RBF backbone without changing its public legacy `forward` function.
- Condition context includes the same sensor tokens, latent encoding/re-injection,
  global summary, sensor readback/refinement, and GL-residual importance bias.
- Cache levels are `none`, `geometry`, and inference-only FP32
  `static_features`. Geometry caches exact top-k indices/distances; static cache
  stores coordinate features, local RBF condition, and coordinate-based latent
  readout.
- Added `legacy_full` and `cached_streamed` sample modes. Cached streaming runs
  point encoding, global/local fusion, coarse scaffold, residual head, and
  Euler/Heun updates entirely inside each query chunk.
- Endpoint maps remain full pointwise maps and are sliced per query chunk;
  default hard and final clamping retain the existing scatter semantics.
- Static and geometry caches are preallocated and filled by chunk, avoiding
  list-concatenate duplication during million-point cache construction.
- The trainer/helper/YAML path exposes execution mode, query chunk size, and
  cache level; the old sample defaults remain legacy-compatible.

### Evidence

- Focused equivalence suite: **21 passed in 10.15 s**.
- Complete regression suite on physical GPU 0: **45 passed in 11.35 s**.
- Both requested gather modes, both solvers, NFE/step counts 1/2/4, every
  observation-consistency mode, and all cache levels match within tight FP32
  tolerance.
- A four-step Heun test asserts observation encoding happens exactly once.
- Real checkpoint/snapshot, Heun-2, endpoint-smooth: maximum absolute difference
  `3.09944e-6`, mean `2.48932e-7`, and identical relative L2
  (`0.7297682166`).
- At 250k queries and Euler-2, cached streaming is **6.70x faster** (`0.671 s`
  vs `4.494 s`) and peak allocation is **69.4% lower** (`852.4 MB` vs
  `2,782.0 MB`).
- One million queries complete in `2.675 s`, or `1.34 s` per million points per
  NFE, at `2,958.4 MB` peak. The explicit FP32 static query cache is
  `2,197.3 MB`; dynamic execution remains bounded by the 8,192-query chunk plus
  state/model workspace.
- Raw CSV/JSON, exact command, limitations, and interpretation are under
  `_CheckNotes/Stage4_reconstruction/`.

### Stage-4 gate decision: PASSED

- Cached-streamed reconstruction is tightly equivalent across the specified
  solver/gather/consistency matrix and on one real checkpoint snapshot.
- Static condition encoding is reused across the full ODE trajectory.
- The 1M-query path is memory-safe and allocates no full dynamic hidden head
  fields; peak scaling is explained by the explicit selected cache level.
- Legacy reconstruction remains callable for A/B validation.

## Stage 5 log

### Implementation

- Added RF bridge primitives that sample one full coherent RFF source field and
  one time per physical sample for the whole effective query set.
- Added query-microbatched loss execution with exact scalar-element weighting.
  The differentiable Stage-4 condition context is built once, retained only as
  needed across chunk backward calls, and never detached.
- Standard training clears gradients once, accumulates every weighted query
  chunk, clips once, and performs one optimizer step per physical DataLoader
  batch. Validation uses the same weighted path under `no_grad`.
- `n_query_points` remains the effective supervision count. New execution keys
  are `train_query_microbatch_size` and
  `reuse_condition_context_across_query_microbatches`.
- The historical monolithic `training_loss()` is unchanged and remains the path
  when the microbatch is null or not smaller than the effective query count.
- Added diagnostic phases for RF bridge, condition context, chunk forward,
  chunk backward, and chunk count.

### Equivalence and scaling evidence

- Complete regression suite on physical GPU 0: **47 passed in 11.69 s**.
- Mandatory effective-31/microbatch-7 test confirms the last 3-point chunk,
  total loss, every gradient, learnable RBF sigma gradient, one clipped Adam
  update, and validation loss. The recording RFF prior is called exactly once
  with all 31 coordinates.
- Raw gradients pass `rtol=8e-5, atol=2e-7`; the largest one-step Adam parameter
  delta is `5.8e-6`, caused by Adam amplification of ~1e-9 FP32 cancellation in
  near-zero attention-bias gradients.
- At 65,536 effective queries, microbatch 4,096 reduces peak allocation from
  `3,025.9 MB` to `323.7 MB` (**89.3%**) for a 26.5% time cost. Microbatch 8,192
  uses `513.0 MB` (**83.0% lower**) for an 11.5% cost.
- A 4,096-query chunk uses `320.9 MB` at 16,384 effective queries and
  `323.7 MB` at 65,536, demonstrating that activation peak follows chunk size,
  not total supervision.
- Matched benchmark mean loss at 65,536 is `2.069149295` monolithic,
  `2.069149295` micro-4k, and `2.069149335` micro-8k.
- Raw CSV/JSON, exact command, phase timings, and interpretation are under
  `_CheckNotes/Stage5_query_microbatch/`.

### Stage-5 gate decision: PASSED

- RF stochastic coherence and the unchanged mean-MSE objective are preserved.
- Losses, all gradients, learnable sigma, and one optimizer update are
  FP32-equivalent.
- One physical batch still produces one clipped optimizer step.
- 65,536 effective queries train successfully, and peak activation memory is
  controlled by microbatch size.

## Limited validation log

- Package: `_CheckNotes/Stage1_5_limited_run/` with two configs, launcher,
  checkpoint-reconstruction recovery utility, analyzer, compact evidence, and
  README.
- Control A: 8 completed epochs, 4,096 monolithic queries, final train/validation
  `1.044623 / 1.038733`, fixed-manifest mean `0.948706`, sampled peak
  `20,745.5 MB`, steady logged epoch time `58.72 s`.
- Large-effective-query B: 5 completed epochs, 16,384 effective / 4,096 execution
  queries, final train/validation `1.188536 / 1.197108`, fixed-manifest mean
  `1.053499`, sampled peak `26,267.3 MB`, steady epoch time `108.69 s`.
- A/B losses are nearly identical through epoch 4. B is 3.9% higher at epoch 5
  but remains stable; the fixed-manifest values are not a same-epoch accuracy
  comparison (A checkpoint epoch 8, B epoch 5).
- Both final checkpoints completed cached Euler NFE 1/2 reconstruction with 256
  hard observations and zero sensor-consistency error.
- The Stage-4 one-million-query Euler-2 stress is integrated into the package
  summary: `2.675 s`, `2,958.4 MB` peak, `2,197.3 MB` explicit static cache.
- GPU 0 started with a 10,636 MiB/100%-utilization co-tenant, so times remain
  relative diagnostics; process-local CUDA peaks are valid.
- Control exposed one deferred-read boundary bug only after its epoch-8
  checkpoint was saved. `visualize_reconstruction()` now uses
  `get_full_snapshot()` when provided. Its final reconstruction was recovered
  from the saved checkpoint without retraining; B completed normally afterward.

### Limited validation decision: PASSED

- Revised standard training, microbatched training, validation, checkpointing,
  cached reconstruction, fixed-manifest evaluation, and million-query stress all
  complete within the intended limited scope.
- No long formal training or Stage-6 experiment was launched.

## Final audit and Stage-6 readiness

### New execution/configuration controls

- `field_normalization_mode`: `legacy_full_after_read` or
  `selected_after_full_read`.
- `data_path_diag_storage_mode`: `legacy_rewrite` or `append`.
- `reconstruction_execution_mode`: `legacy_full` or `cached_streamed`.
- `reconstruction_query_chunk_size`: positive query execution chunk.
- `reconstruction_cache_level`: `none`, `geometry`, or `static_features`.
- `train_query_microbatch_size`: null/large for monolithic, or a smaller positive
  execution chunk for effective-query microbatching.
- `reuse_condition_context_across_query_microbatches`: boolean differentiable
  context reuse.

### Legacy/reference paths retained

- Complete legacy data-path profile and its component overrides.
- Full-after-read normalization and diagnostic history rewrite.
- Indexed-union HDF5 experiment (not default).
- Original GL-RBF `forward()` and `PointCloudFFM.training_loss()`.
- `legacy_full` reconstruction with original Euler/Heun loop.
- Null/no training query microbatching.

### Main changed areas

- Core execution: `src/pointcloud_data_path.py`, `src/Model.py`,
  `src/train_pointcloud_ffm.py`, and `src/helpers.py`.
- Reproducibility/benchmarks: fixed-manifest generator/evaluator plus data-path,
  scaling, reconstruction, and query-microbatch benchmark utilities under
  `src/`.
- Regression coverage: data-path, manifest, scaling schema, reconstruction
  equivalence, and all-gradient query-microbatch tests under `tests/`.
- Evidence/configuration: `Save_config/config_pointcloud_ffm.yaml` and staged
  reports/configs/raw compact results under `_CheckNotes/`.

### Remaining bottlenecks and limitations

- Per-query model execution and activation memory are the dominant scaling cost;
  exact top-k work rises with observation count even under KeOps.
- FP32 `static_features` reconstruction cache is linear in query count
  (`2,197.3 MB` at 1M). `geometry`/`none` trade cache memory for repeated work.
- Microbatching trades 11–27% wall time for much lower activation memory in the
  tested range.
- The repository has no formal 250k/1M 3-D HDF5 dataset. Large data-path rows use
  a labeled in-memory expansion, so future storage-layout/I/O conclusions need
  the formal dataset.
- GPU wall times were collected with a documented co-tenant and should be rerun
  exclusively before publication-grade claims.
- The limited 5/8-epoch runs validate execution and short-horizon stability, not
  final scientific accuracy.

### Invariant confirmation

- No GL-RBF parameterization, learned weight shape, or architecture was changed.
- No top-k/KeOps neighbor or gather mathematics was changed.
- No Rectified-Flow convention, RFF distribution, `x1-x0` target, or mean-MSE
  objective was changed.
- No Euler/Heun formula, observation meaning, or sensor-consistency rule was
  changed.
- Existing checkpoint parameters load strictly and are not reinterpreted.
- No Stage-6 architecture experiment was started.

### Recommendation

**Ready to proceed to a separately approved Stage 6.** All readiness criteria
are met: the data path is validated, matched evaluation shows no material
regression, cached-streamed reconstruction is equivalent and 1M-query safe,
microbatch gradients/updates are equivalent, and 65,536-query supervision is
practical. Profiling now identifies genuine per-query model architecture as the
dominant remaining cost. Before making publication-grade performance claims,
repeat key timings on an exclusive GPU and on the formal large 3-D HDF5 layout.

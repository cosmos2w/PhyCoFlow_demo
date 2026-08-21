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
| 2. Data-path polish | Not started | Pending |
| 3. Million-point scaling | Not started | Pending |
| 4. Cached/streamed reconstruction | Not started | Pending |
| 5. Query-microbatch training | Not started | Pending |
| Limited validation package | Not started | Pending |

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

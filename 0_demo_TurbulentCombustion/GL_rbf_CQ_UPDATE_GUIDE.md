# GL_rbf_CQ portable update guide for AI coding agents

This guide is an execution contract for migrating an older DMF-Gen or
PointCloud FFM project whose dataset, fields, loaders, and training orchestration
differ from the turbulent-combustion demo. Preserve the downstream data stack;
replace only the reusable model boundary described here.

## Source and frozen release identity

- Source development branch: `perf/stage8-attention-optimization`
- Source Stage-8 tip: `7b5dd2df672a86d5b5cf2502b93208786d282e17`
- Stage-8 code tag: `gl-rbf-cq-v0.9.0-rc2` at
  `f8d57153c268a0e03c941092dffc921898d58d61`
- Portable-preparation branch: `release/gl-rbf-cq-portable-prep`
- Public model name: `GL_rbf_CQ`
- Historical config/class name: `GL_rbf_ENH_CQ`
- Release weights: unchanged Stage-7 epoch-1000 EMA-resolved portable checkpoint
- Checkpoint SHA256:
  `2516ffeb45775d4e6b8d88b4b24d927aac28665a2a90102583e07deaca78f64d`

Do not infer that the RC2 tag contains later repository-interface cleanup. The
exact reusable source baseline for this migration is the Stage-8 tip SHA above.

## Accepted Stage 1–8 chain

The migration target is cumulative. Do not selectively recreate old rejected
experiments.

1. Stage 1 established measured PointCloud FFM data/model cost boundaries.
2. Stage 2 accepted the optimized selected-query data path while retaining the
   historical data path for controlled reproduction.
3. Stage 3 established 4K/16K/65K scaling and explicit memory accounting.
4. Stage 4 accepted cached, streamed reconstruction with reusable condition
   context.
5. Stage 5 accepted mathematically equivalent query microbatching.
6. Stage 6 accepted the 128-D low-rank compact-query decoder and persistent
   geometry-only Top-K with K=32.
7. Stage 7 froze the balanced scientific architecture: latent width 256,
   128-D CQ decoder, sinusoidal FiLM, raw measurement/support shortcut, and EMA.
8. Stage 8 changed execution only: cached sensor K/V with full padding is the
   preferred path. `legacy_mha` remains the compatibility/debug path.

Static sensor bucketing, dynamic trimming, structured-concat sweeps, SDPA model
redesigns, fused optimizer changes, and new losses are not release defaults.

## What to migrate

Use `GL_rbf_CQ_RELEASE_MANIFEST.yaml` as the machine-readable authority.

| Classification | Components | Action |
|---|---|---|
| Required portable core | `src/phycoflow_pointcloud/` files listed in the manifest | Copy as one package and preserve relative layout |
| Required model defaults | `configs/gl_rbf_cq_core.yaml` | Merge with the downstream config; do not use it as a dataset config |
| Optional acceleration | `pykeops` | Install for `neighbor_backend: keops`, otherwise explicitly use `torch` |
| Compatibility-only | `Model.py`, `model_ema.py`, `obs_consistency.py`, `persistent_topk_geometry_cache.py` shims | Retain only if the downstream project imports historical paths |
| Demo-specific reference | turbulent-combustion train/reconstruct/evaluate entry points and full `gl_rbf_cq.yaml` | Read as examples; do not copy blindly |
| Do not migrate | `_legacy_model_full.py`, `_CheckNotes/`, `research/`, figures, HDF5 loader, coherence/baseline code | Leave outside the downstream runtime |

The package source of truth is
`phycoflow_pointcloud.models.portable_core`, not the monolithic historical
`Model.py`.

The lightweight `Model.py` shim serves GL-RBF/CQ directly. It lazily references
`_legacy_model_full.py` only for obsolete MLP/Perceiver/FNO symbols. A GL-only
downstream project does not need that legacy baseline file or `neuralop`.

## Portable snapshot integrity

The manifest-declared portable core is a byte-level vendored snapshot, not a
source tree to reformat. Copy every `portable_core_files` entry at its exact
relative path, then compare each source/destination with `cmp` and SHA256
before importing it; record the manifest hash and the per-file hashes in the
migration evidence. A missing or changed byte is a failed gate. Exclude the
vendored `src/phycoflow_pointcloud` snapshot from Ruff, Black, and other
rewrite/format tools, while continuing to lint and format the downstream
adapter and integration code.

## Tensor-level integration contract

The downstream project owns loading, normalization, field order, sampling, and
serialization. It must provide:

| Tensor | Shape | Meaning |
|---|---|---|
| `coords` | `[B, N, D]` | query coordinates |
| `x_t` | `[B, N, F]` | RF state at time `t` |
| `obs_coords` | `[B, M, D]` | padded sparse-sensor coordinates |
| `obs_values` | `[B, M, 1]` | one scalar measurement per sensor slot |
| `obs_mask` | `[B, M]` | 1/True for valid sensors, 0/False for padding |
| `obs_field_ids` | `[B, M]` | integer channel ID in downstream field order |
| `t` | `[B]` | rectified-flow time |
| training target `x1` | `[B, N, F]` | normalized clean field state |
| optional `obs_indices` | `[B, M]` | query indices used for exact sensor clamping |

Rules:

- `D` may be 2 or 3; set `coord_dim` accordingly before model construction.
- `F` is supplied as `n_fields` to `build_pointcloud_model` and is not fixed to
  five.
- All coordinate tensors must share device, dtype, and coordinate convention.
- Field order and normalization statistics belong to the downstream project.
- Sensor and query sampling belong to the downstream project.
- Masked sensor slots must have `obs_mask=0`; their values and IDs are ignored.
- Rebuild persistent geometry whenever query coordinates, sensor coordinates,
  mask, shape, storage identity/version, device, or dtype changes.
- Sensor values may change while reusing a valid geometry cache.

The core has no field-name table and no HDF5 assumption.

## Semantic checkpoint and data identity

Tensor shapes and state-dict keys are necessary but not sufficient for loading
release weights. Before calling `load_state_dict(..., strict=True)`, validate
the semantic contract from checkpoint metadata and the downstream dataset:

- Compare the exact ordered field identity list (`field_names` in the release
  checkpoint or `data_spec.field_names` in a native checkpoint) with the
  downstream `DataSpec.field_names`, and verify `n_fields`. Equal channel
  counts do not make a different name or order safe; reject that checkpoint
  before loading any state tensor.
- Compare the normalization contract, including method, training-split
  provenance/dataset fingerprint when present, and the exact per-field
  offsets/scales or digest. The release checkpoint's `mean`/`std` metadata
  must match the downstream verified normalizer; missing semantic metadata is
  not evidence of compatibility.
- Also verify model/backbone identity, coordinate convention, and the
  expected state schema/checksum. These checks precede the strict tensor load.

On any semantic mismatch, reject the release checkpoint; do not permute
channels, remap keys, or use `strict=False`. A documented fresh-start
downstream benchmark is allowed when the downstream field schema or
normalization intentionally differs: construct the model from that downstream
contract and train from scratch. The portable package and state-schema gates
remain strict, and execution-variant arms (for example B/C) must still use
the same seed/config and pass their matched seeded-initialization identity
check before training.

## Old-to-new mapping

| Older project symbol/config | Portable target |
|---|---|
| `Model.ConditionalPointHybridLocalGlobalRBFCQ` | `phycoflow_pointcloud.GL_rbf_CQ` |
| `Model.ConditionalPointHybridLocalGlobalRBF` | `phycoflow_pointcloud.GL_rbf_ENH` or core class |
| `Model.PointCloudFFM` | `phycoflow_pointcloud.PointCloudFFM` |
| hand-built model constructor | `phycoflow_pointcloud.build_pointcloud_model(config, n_fields=F)` |
| `model_ema.ModelEMA` | `phycoflow_pointcloud.ModelEMA` |
| top-level persistent cache module | `phycoflow_pointcloud.cache` |
| old CQ config missing Stage-8 keys | remains `legacy_mha + full` for reproduction |
| new GL_rbf_CQ run | explicitly `cached_kv + full` |

Historical checkpoint keys are unchanged. Do not write a key-renaming migration.

## Generic trainer and evaluator lifecycle

Keep lifecycle behavior model-agnostic; register optional hooks instead of
branching on a model name. For an adapter that supports query microbatching:

- The generic trainer calls
  `training_backward(batch, *, loss_scale, start_phase=None, end_phase=None)`
  when present. The adapter owns the bridge, condition-context preparation,
  and query-chunk forward/backward loop; when configured, it reuses one
  condition context across query chunks, weights each chunk by the total
  element count, and performs exactly one backward per chunk. Phase callbacks
  must bracket bridge/context preparation and each forward/backward segment;
  the trainer must not call `training_loss().backward()` a second time.
- `loss_scale` is applied exactly once. The generic trainer clears gradients,
  checks finite loss/gradients, and clips; the adapter must explicitly reject
  unsupported scale or adaptive-scaling requests. The B/C contract is
  `loss_scale=1` with adaptive scaling disabled.
- After a successful optimizer update, call `after_optimizer_step()`. Put
  resumable EMA state in `training_aux_state_dict()` and restore it through
  `load_training_aux_state_dict(state)` with strict presence/schema checks.
  The evaluator uses `evaluation_weight_context()` for configured weights;
  expose an explicit `configured` versus `live` selection, where `live`
  bypasses EMA and `configured` follows the model's EMA-evaluation setting.
  Do not special-case `GL_rbf_CQ` (or any other model name) in the trainer or
  evaluator.

## Minimal config patch

Start by merging `configs/gl_rbf_cq_core.yaml` into the downstream model section,
then set only downstream-owned values. At minimum verify:

```yaml
model_name: GL_rbf_CQ
backbone: GL_rbf_ENH_CQ
coord_dim: 2                 # or 3 for the downstream geometry

latent_dim: 256
cq_query_dim: 128
cq_readout_mode: lowrank
cq_readout_rank: 64
cq_readout_heads: 4
cq_fusion_mode: additive
gather_mode: topk_rbf_glres
gather_topk: 32

condition_attention_execution: cached_kv
sensor_attention_padding_mode: full
```

Do not copy `data`, `save_dir`, field names, conditioned fields, observation
counts, split ratios, batch size, or demo identifiers from the combustion YAML.

## Migration workflow for another AI agent

1. Freeze the downstream branch, current checkpoint hashes, config, and one
   deterministic synthetic forward/reconstruction fixture.
2. Inventory downstream imports of `Model`, EMA, observation consistency, and
   persistent-cache helpers. Do not change the loader yet.
3. Copy the manifest-declared portable package as a unit. Do not copy only
   `gl_rbf_cq.py`; it depends on the package-local attention, cache, prior,
   observation, checkpoint, and RF components. Verify byte identity and
   checksums for every copied manifest file, and configure formatters/linters
   to exclude the vendored snapshot.
4. Add `gl_rbf_cq_core.yaml` and implement an explicit config merge where
   downstream dataset/training values override model defaults.
5. Replace model construction with
   `build_pointcloud_model(merged_config, n_fields=len(downstream_fields))`.
6. Adapt the existing batch into the tensor contract above. Keep the existing
   field order and normalization exactly unchanged.
7. Before any checkpoint load, apply the semantic field-identity and
   normalization checks above. Stop on a same-count field reorder/name change,
   normalization mismatch, or missing identity metadata.
8. If the semantic gate passes, strict-load the checkpoint and stop on any
   missing or unexpected state keys; do not add a permissive loader. If the
   downstream contract intentionally differs, record the rejected release
   checkpoint and use the documented fresh-start path instead.
9. For a compatible checkpoint, compare old/new seeded forward output and
   gradients in FP32 using the same tensors and loss. For a fresh-start
   execution comparison, verify the matched B/C seeded initial state first.
10. Compare monolithic and query-microbatched RF loss/gradients using one
    shared random bridge seed and the exact configured loss scale.
11. Compare legacy and cached-streamed reconstruction with identical prior RNG,
    solver, NFE, observation mode, and query tensors.
12. Build persistent Top-K once, instrument `_get_topk_neighbors`, and prove
    zero additional KNN calls during cached reconstruction.
13. Only after all gates pass should the downstream training/evaluation wrapper
    select the new public model name. Keep a `legacy_mha` debug override.

## Required verification gates

- Isolated package import/build with no `Model.py`, `helpers.py`, dataset,
  baseline, visualization, or coherence modules available.
- Synthetic 2-D and 3-D forward and backward.
- At least two supported field counts, including the downstream count.
- Mixed valid sensor counts in one padded batch.
- Cached-K/V execution and `legacy_mha` diagnostic construction.
- Byte-identical/checksum-verified portable snapshot with formatters excluded.
- Semantic field identity/order and normalization validation before strict load;
  equal `n_fields` with different positional meaning must fail explicitly.
- Strict checkpoint load and identical state schema when the semantic contract
  matches, or a recorded fresh-start gate when it intentionally does not.
- Seeded old/new forward, loss, every parameter gradient, and reconstruction
  within the established RC tolerances.
- Monolithic/query-microbatch RF equivalence.
- Persistent geometry/static-feature equivalence and zero post-build KNN calls.
- Exact loss-scale behavior, query-condition reuse, and no-double-backward
  lifecycle integration.
- EMA save/load/resume and configured/live evaluation selection without
  model-name branching.
- Matched seeded B/C initialization identity for execution-variant benchmarks.
- Full downstream regression suite.

Record torch, CUDA, KeOps, dtype, device, seeds, and solver/NFE in evidence.

## Explicit do-not-do list

- Do not retrain to hide a migration mismatch.
- Do not reuse release weights after a field-identity/order or normalization
  mismatch, even when `n_fields` and tensor shapes match; use the documented
  fresh-start path and record the reason.
- Do not rename state-dict keys, permute channels, or load with `strict=False`.
- Do not edit, format, or rewrite the byte-identical vendored portable snapshot;
  exclude it from downstream rewrite tools.
- Do not widen CQ, change latent count/blocks, K, sigma, FiLM, measurement/support,
  RF objective, prior, EMA semantics, or solver behavior.
- Do not make condition latents or static caches time-dependent.
- Do not run a second KNN search for measurement/support features.
- Do not promote static buckets or dynamic trimming.
- Do not copy the combustion HDF5 loader, field names, normalization, sampling,
  visualization, or research scripts into a different project by default.
- Do not apply model-name branches in generic trainer/evaluator lifecycle code;
  use the optional hook contract and explicit configured/live selection.
- Do not mix optimizer/kernel experiments into the portability comparison.
- Do not modify the downstream project until its own migration task begins.

## Prepared downstream validation: Proj_MultiFieldReconstruction

`Proj_MultiFieldReconstruction/` is intentionally untouched in this task. In
the next task, treat it as an independent consumer: give its agent this guide
and the release manifest, preserve its dataset/field pipeline, and evaluate
whether the migration can be completed using only the tensor contract and
portable package. Any missing instruction found there should be fixed in this
guide before preparing a release PR for `main`.

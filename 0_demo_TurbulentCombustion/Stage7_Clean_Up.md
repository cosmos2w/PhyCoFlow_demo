# Stage 7 post-RC cleanup plan

Status: **Phases 0–5 executed on `cleanup/gl-rbf-cq-rc1`; Phase 6 deferred**  
Compatibility baseline: annotated tag `gl-rbf-cq-v0.9.0-rc1`  
Public target names: `GL_rbf_CQ`, `GL_rbf_CQ-fast`, `GL_rbf_ENH` (legacy)

Execution evidence is indexed in `_CheckNotes/Stage7_cleanup/RESULTS.md`.

## 1. Purpose and guardrails

The RC1 code is scientifically validated but organized as an experiment-grown
research tree. The cleanup should make the balanced CQ model understandable and
runnable by a collaborator without changing numerical behavior, checkpoint
loading, persistent-cache semantics, or the accepted scientific configuration.

This document defines the work. It does not rename, move, delete, or refactor
anything in the RC freeze.

Hard guardrails for the later cleanup revision:

1. branch from `gl-rbf-cq-v0.9.0-rc1`, never from an unrecorded worktree;
2. preserve the tag as the byte-for-byte compatibility oracle;
3. do not retrain to validate refactoring;
4. require old/new numerical comparison on the same checkpoints and inputs;
5. keep the internal `GL_rbf_ENH_CQ` and historical state keys loadable;
6. keep Top-K indices/distances, K=32, KeOps, sigma, GLRES, RF objective,
   optimized data path, and scheduler semantics unchanged;
7. do not combine cleanup with SDPA, optimizer, architecture, or objective
   changes; and
8. treat removal as a separately reviewed phase after a retention manifest and
   recoverable archive exist.

## 2. Current-tree audit

### 2.1 Runtime-critical source

| Current path | Size/role | Cleanup issue |
|---|---|---|
| `src/Model.py` | 3,706 lines; GL-RBF core, CQ subclass, PointCloudFFM, priors/readouts | Multiple model layers and compatibility branches are concentrated in one file. |
| `src/train_pointcloud_ffm.py` | 2,624 lines; config parsing, training, validation, checkpointing, EMA, microbatching | Training orchestration, model factory, persistence, evaluation, and logging are intertwined. |
| `src/helpers.py` | 1,230 lines; dataset, sparse condition, visualization/reconstruction | Dataset access and visual/report code share a module. |
| `src/model_ema.py` | 101 lines; trainable-only EMA and legacy-shadow repair support | Small and cohesive; should move with tests mostly unchanged. |
| `src/persistent_topk_geometry_cache.py` | persistent geometry storage/load | Runtime-critical and should remain isolated. |
| `src/pointcloud_data_path.py` | optimized data path | Runtime-critical; needs a portable dataset-path boundary. |
| `src/obs_consistency.py` | observation constraints | Runtime-critical but independent of architecture. |
| `src/pointcloud_eval_manifest.py` | fixed-manifest protocol | Evaluation-critical; should become a stable protocol module. |
| `src/evaluate_pointcloud_fixed_manifest.py` | model build and controlled RF evaluation | Duplicates model/config construction used in training and other evaluators. |

Key class locations in the RC:

- `ConditionalPointHybridLocalGlobalRBF` at `src/Model.py:716`;
- `ConditionalPointHybridLocalGlobalRBFCQ` at `src/Model.py:1892`;
- `PointCloudFFM` at `src/Model.py:2983`; and
- `ModelEMA` at `src/model_ema.py:11`.

### 2.2 Research and benchmark scripts

The top-level `src/` directory includes permanent runtime code beside one-off or
stage-specific entry points:

- `benchmark_pointcloud_data_path.py`;
- `benchmark_pointcloud_scaling.py`;
- `benchmark_pointcloud_reconstruction.py`;
- `benchmark_pointcloud_query_microbatch.py`;
- `benchmark_pointcloud_cq.py`;
- `benchmark_cq_persistent_topk_cache.py`;
- `benchmark_pointcloud_stage7.py`;
- `analyze_stage7_screen.py`; and
- several legacy/general evaluation scripts.

These are valuable provenance but should not define the collaborator-facing
runtime package. They should move to `research/benchmarks/` or
`research/stages/` only after imports and exact commands are updated and tested.

### 2.3 Tests

The RC has focused coverage across eleven top-level test modules:

- data path and scaling;
- reconstruction streaming;
- query microbatching;
- CQ decoder and CQ-Balanced compatibility;
- persistent Top-K;
- fixed-manifest evaluation;
- Stage 7 EMA/FiLM/measurement support; and
- Stage 7 screen analysis.

This is a strong starting point. The cleanup should reorganize tests by public
contract rather than delete them as “stage-specific.” A test is obsolete only
when its behavior is covered by a clearer compatibility or integration test.

### 2.4 Configuration sprawl

`Save_config/pointcloud_ffm/` contains many timestamped `DemoN*` YAML snapshots
and an additional `bk/` tree. `_CheckNotes/Stage6_*` and `_CheckNotes/Stage7_*`
contain more exact-run YAMLs. Some active and historical YAMLs include absolute
`/home/wanglz/...` paths. This is useful provenance but hostile to portable use.

The cleanup should create three small public configs while preserving exact-run
YAMLs in a research archive:

- `configs/gl_rbf_cq.yaml`;
- `configs/gl_rbf_cq_fast.yaml`; and
- `configs/legacy_gl_rbf_enh.yaml`.

Dataset and output roots should be CLI/config parameters or environment inputs,
not developer-specific absolute paths.

### 2.5 Reports, generated evidence, and binary artifacts

Current local footprint is approximately:

| Area | Local size | Classification |
|---|---:|---|
| `_CheckNotes/` | 3.5 GiB | mixed canonical reports and ignored run products |
| `figures/` | 92 MiB | scripts, source tables, vectors, previews, TIFFs |
| `Save_TrainedModel/` | 40 GiB | historical binary artifacts; not source control |
| `ReleaseArtifacts/` | 168 MiB | selected RC research checkpoints/stats/config |
| `.worktrees/` | 2.1 GiB | linked experimental worktrees and ignored evidence |

Canonical reports must remain tracked. Large checkpoints/logs/reconstruction
intermediates should be managed by an artifact manifest and storage policy, not
mixed with source status. The RC artifact manifest is the initial model for
this separation.

### 2.6 Older versions and legacy baselines

`src/_OlderVersion/` includes prior helpers and three prior PointCloud training
scripts as well as FNO/Senseiver code. These files should not be deleted merely
because a current path exists. First determine whether they are:

- required to load or reproduce a published/archived checkpoint;
- a reference implementation with unique semantics;
- already preserved by Git tag and durable artifact metadata; or
- genuinely redundant.

Senseiver and Latent FM remain reference baselines. Their runtime code is not to
be rewritten as part of GL_rbf_CQ cleanup.

## 3. Proposed collaborator-facing structure

```text
0_demo_TurbulentCombustion/
├── README.md
├── ModelUpdate.md
├── configs/
│   ├── gl_rbf_cq.yaml
│   ├── gl_rbf_cq_fast.yaml
│   └── legacy_gl_rbf_enh.yaml
├── src/
│   └── phycoflow_pointcloud/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── conditions.py
│       ├── priors.py
│       ├── training.py
│       ├── checkpointing.py
│       ├── reconstruction.py
│       ├── evaluation.py
│       ├── cache/
│       │   ├── geometry.py
│       │   └── static_features.py
│       └── models/
│           ├── gl_rbf_core.py
│           ├── gl_rbf_enh.py
│           ├── gl_rbf_cq.py
│           ├── query_readout.py
│           ├── timestep.py
│           └── ema.py
├── scripts/
│   ├── reconstruct_pointcloud.py
│   ├── evaluate_fixed_manifest.py
│   └── benchmark_pointcloud.py
├── tests/
│   ├── compatibility/
│   ├── unit/
│   ├── equivalence/
│   └── integration/
├── research/
│   ├── stages/
│   ├── benchmarks/
│   ├── rejected/
│   └── archived_configs/
├── figures/
│   ├── scripts/
│   └── generated/
└── artifacts/
    └── MANIFEST.md
```

This is a target topology, not a mandate to move every file in one commit.
Small, behavior-preserving slices are safer.

## 4. Public API and naming plan

### 4.1 Public names

Expose a small model factory:

```python
model = build_pointcloud_model("GL_rbf_CQ", config)
model = build_pointcloud_model("GL_rbf_CQ-fast", config)
model = build_pointcloud_model("GL_rbf_ENH", config)
```

The factory should map to frozen internal classes and state keys. Do not rename
checkpoint parameters. A public alias is sufficient.

### 4.2 Compatibility aliases

The loader must continue accepting:

- `backbone: GL_rbf_ENH_CQ`;
- the historical F0 identifiers;
- configs with none of the Stage 7 keys;
- live-only checkpoints;
- live-plus-EMA Stage 7 checkpoints; and
- the legacy Stage 7 EMA shadow requiring frozen-state repair.

Unknown or conflicting public/internal names should fail with a precise error,
not silently select a different architecture.

### 4.3 Minimal commands

The intended user flow is:

```bash
python src/train_pointcloud_ffm.py --config configs/gl_rbf_cq.yaml
python scripts/reconstruct_pointcloud.py \
  --config configs/gl_rbf_cq.yaml \
  --checkpoint /path/to/checkpoint.pt
python scripts/evaluate_fixed_manifest.py \
  --config configs/gl_rbf_cq.yaml \
  --checkpoint /path/to/checkpoint.pt
```

Advanced Stage 1–7 flags remain overrideable, but a collaborator should not
need to understand the history to run the default.

## 5. Source decomposition plan

### Slice A — config schema without model changes

Create typed/defaulted parsing for the three public presets. Normalize paths at
the CLI boundary. Keep the current model factory underneath. Golden-test the
fully resolved config dictionary against RC1.

### Slice B — checkpoint and EMA boundary

Move checkpoint selection, EMA state interpretation, and legacy frozen-state
repair into `checkpointing.py`. Add an explicit result object that states
`live`, `ema`, or `ema_trainable_plus_live_frozen`. Preserve exact tensors.

### Slice C — geometry/static caches

Move persistent Top-K construction and cache-level selection behind a small
interface. Keep the existing cache payload schema and add version metadata only
in a backward-compatible wrapper. Assert zero search calls after build.

### Slice D — GL-RBF condition core

Extract sensor embedding, latent blocks, reverse sensor readout, local RBF
gather, and GLRES weighting from `Model.py` without altering module names/state
paths. The safest first step is re-exporting or composition with compatibility
properties, not renaming submodules.

### Slice E — CQ query decoder

Extract point encoding, low-rank readout, additive/structured compatibility,
FiLM, raw measurement/support, and compact field head. Keep rejected
structured-concat code in a research compatibility module until checkpoint/key
search proves no retained artifact depends on it.

### Slice F — training loop

Separate RF bridge/prior sampling, condition reuse, query microbatch reduction,
optimizer/scheduler, validation, and diagnostics. Preserve RNG call order and
the single full-query prior draw; changing either can invalidate equivalence
despite identical formulas.

### Slice G — reconstruction and evaluation

Unify model construction and checkpoint loading across fixed-manifest and
reconstruction scripts. Keep controlled seeds/checksums first-class in output.
Avoid a “generic evaluator” that hides protocol differences.

## 6. Retain, archive, and candidate-removal classification

### Retain in production support

- `GL_rbf_CQ` balanced profile;
- `GL_rbf_CQ-fast` throughput profile;
- `GL_rbf_ENH` load/reproduction profile;
- optimized data path;
- query-microbatch training;
- cached-streamed inference;
- persistent Top-K geometry and static-feature cache levels;
- EMA with legacy-shadow repair;
- sinusoidal CQ FiLM;
- raw measurement/support shortcut;
- fixed-manifest evaluation and sparse-condition/RF checksums; and
- observation-consistency modes used by retained checkpoints.

### Archive as research provenance

- Stage 1–7 exact run configs and reports;
- benchmark launchers and analyzers;
- CQ-Balanced 192/224 cost-gate evidence;
- CQ full/readout/rescue screens;
- attention-kernel comparison;
- Senseiver/Latent FM comparison artifacts;
- timestamped configuration history; and
- publication figure source data/scripts.

### Candidate removal after proof and review

- rejected structured-concat implementation paths with no retained checkpoint;
- duplicate stage-specific model factories;
- redundant one-off launch wrappers;
- obsolete PID/GPU-state/monitor helpers;
- duplicate evaluation scripts with identical protocols;
- stale absolute-path copies after portable equivalents and archived originals
  exist; and
- unneeded pycache/runtime logs.

No candidate in this section should be removed in the same commit that first
introduces its replacement.

## 7. Binary artifact and storage plan

1. Keep `_CheckNotes/GL_rbf_CQ_rc1_artifacts.md` as the source inventory.
2. Produce a coworker-portable, EMA-resolved checkpoint in a later release step:
   one selected model state, normalization stats, public config, format version,
   source tag, field order, and SHA-256.
3. Validate that the resolved export reproduces the research checkpoint under
   corrected loading before designating it a release artifact.
4. Store large binaries in an agreed artifact store or Git LFS only after the
   repository policy is selected; do not force-add 88 MB checkpoints ad hoc.
5. Retain live-plus-EMA research checkpoints for resume/audit even if a smaller
   inference export is created.
6. Add a manifest check command that reports missing, mismatched, or unexpected
   files without deleting them.

## 8. Phased implementation and acceptance gates

### Phase 0 — freeze verification

- resolve `gl-rbf-cq-v0.9.0-rc1` and verify source/doc SHA;
- verify all artifact SHA-256 values;
- run the full RC test suite;
- materialize fixed deterministic input bundles for old/new comparison; and
- record Python/PyTorch/CUDA/KeOps versions.

Gate: no cleanup branch work begins if tag or artifacts cannot be reproduced.

### Phase 1 — public configs and read-only factory

- add three portable configs;
- add public aliases that call frozen builders;
- add CLI path overrides;
- do not move model code yet.

Gate: resolved configurations and model state dictionaries match RC1 exactly.

### Phase 2 — checkpoint/EMA and cache modules

- isolate checkpoint semantics;
- isolate persistent geometry/static-cache API;
- keep compatibility imports in old locations.

Gate: checkpoint round-trip, legacy load, EMA repair, cache serialization, and
zero-extra-KNN tests pass; matched outputs meet RC tolerances.

### Phase 3 — model decomposition

- extract condition core and CQ query decoder in small commits;
- preserve module/state names via aliases or explicit key translation;
- compare state-key set and tensor values after every slice.

Gate: forward, loss, gradients, optimizer update, monolithic/microbatch, and
cached/streamed equivalence pass for F0, CQ-fast, and CQ-balanced default.

### Phase 4 — training/evaluation entry points

- introduce the three clean commands;
- reuse one model/checkpoint builder;
- preserve seed and RNG order;
- emit protocol and checksum metadata.

Gate: a short deterministic smoke/resume cycle matches the frozen entry points;
no long retraining is required.

### Phase 5 — research archive and deprecation

- move research scripts/configs with a machine-readable mapping;
- add deprecation shims for old imports/commands;
- update links in tracked reports;
- mark candidate removals but do not remove until a later release.

Gate: every canonical Stage 1–7 report link resolves, old commands either work
or fail with migration instructions, and Git history/tag provides recovery.

### Phase 6 — optional removals

- perform only reviewed removals from the candidate list;
- report exact deleted paths, archive/tag recovery, and reclaimed size;
- never remove user run directories by glob.

Gate: full suite plus compatibility matrix remains green; release owner signs
off on artifact retention.

## 9. Required compatibility and equivalence matrix

| Area | Required cases | Acceptance |
|---|---|---|
| Config defaults | old F0 YAML, old CQ YAML, public balanced/fast/legacy | Same resolved scientific flags and model shapes. |
| Checkpoint loading | live-only, live+EMA, legacy EMA repair, resume optimizer/scheduler | Strict state load; exact epoch/global step; expected selection semantics. |
| Forward | F0, CQ-fast, GL_rbf_CQ; latent 128/256 | `allclose` at existing test bounds; field order identical. |
| FiLM | disabled scalar path, zero-init identity, enabled gradients | Historical output when disabled; nonzero gradients when trained. |
| Raw measurement/support | hand calculation, absent fields, normalization on/off | Exact supported values; finite zero-support behavior. |
| Top-K | cached/uncached, geometry/static, 4k/1M metadata | Same indices/distances; zero post-build KNN. |
| Reconstruction | Euler/Heun; NFE 1/2/4; consistency modes | Existing max/mean error tolerances and identical protocol metadata. |
| Microbatch | uneven final chunk; validation; clipped Adam update | Existing gradient/update tolerances; one prior call. |
| Fixed manifest | 64 layouts × 3 RF repeats | Same manifest/materialized-input checksum and candidate means within numerical tolerance. |
| Performance | B128/Q4096 and persistent 1M/NFE-4 | No material regression without explicit approval; preserve RC gates. |

## 10. 3-D and field-semantics audit

The cleanup must verify rather than assume dimensional generality. The current
demo uses 2-D coordinates embedded in the turbulent-combustion workflow, while
the scientific model is intended for point clouds more broadly.

Audit checklist:

1. locate every hard-coded coordinate dimension, reshape, meshgrid, plotting
   path, and Fourier-frequency construction;
2. separate model-coordinate dimension from visualization dimension;
3. verify Top-K/KeOps expressions and persistent-cache schemas for 2-D and 3-D;
4. verify that raw measurement/support is independent of coordinate dimension;
5. ensure query microbatch and reconstruction chunking preserve arbitrary
   coordinate channels;
6. add synthetic 3-D forward/cache/microbatch tests without claiming trained
   3-D quality;
7. preserve field order and stats explicitly in checkpoint metadata; and
8. document which visualization utilities remain 2-D-only.

This audit may reveal compatibility work, but it must not silently alter the
frozen turbulent-combustion configuration.

## 11. Documentation migration

- Keep `ModelUpdate.md` as the scientific evolution/reference document.
- Make `README.md` answer installation, data path, three presets, three commands,
  artifact acquisition, and expected hardware first.
- Preserve Stage 1–7 reports under `research/stages/` or `_CheckNotes/archive/`
  with redirect/index links.
- Generate a config reference from the typed schema so defaults do not drift.
- Document cache lifecycle and invalidation explicitly.
- Document checkpoint live/EMA/resolved semantics with examples.
- Keep accepted/rejected scientific decisions in a durable decision table.

## 12. Completion definition for the future cleanup

Cleanup is complete only when:

- a fresh clone plus external artifacts can run the public balanced preset;
- the three public profiles are unambiguous;
- all retained historical checkpoints load;
- the full compatibility/equivalence matrix passes against RC1;
- fixed-manifest and reconstruction checksums/metrics remain within recorded
  numerical tolerances;
- persistent 1M-query inference performs no post-build KNN and does not regress
  materially;
- source modules have clear responsibilities without circular imports;
- exact research provenance remains discoverable;
- removed material has a documented tag/archive recovery path; and
- the normal checkout is clean with no dependency on a hidden worktree.

Until those gates are met, `gl-rbf-cq-v0.9.0-rc1` remains the authoritative
validated implementation.

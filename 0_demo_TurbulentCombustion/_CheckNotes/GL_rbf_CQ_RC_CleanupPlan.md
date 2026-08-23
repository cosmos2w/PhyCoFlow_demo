# Codex Task — Freeze GL_rbf_CQ Release Candidate, Document Model Evolution, and Prepare Cleanup Plan

## Scope

Repository:

`cosmos2w/PhyCoFlow_demo`

Current validated Stage-7 branch:

`perf/pointcloud-smart-cq-stage7`

Current verified Stage-7 tip at the time this task was written:

`a03213ae33f14cbffd198c2da65aa8a11e6b2f26`

Commit message:

`eval(pointcloud): finalize Stage 7 smart CQ`

The current Stage-7 implementation has completed the 1000-epoch formal protocol and is the recommended balanced PointCloud FFM configuration.

Public/release-facing model name from now on:

# `GL_rbf_CQ`

Do **not** call the release model `Stage7-All256`.

The current internal code may still use `GL_rbf_ENH_CQ` / `ConditionalPointHybridLocalGlobalRBFCQ`. Preserve those names in the frozen source snapshot for checkpoint compatibility. The later cleanup plan should explain how to expose `GL_rbf_CQ` as the clean public name without breaking old checkpoints.

This task has exactly three deliverables:

1. freeze and synchronize the validated current revision;
2. create a comprehensive `ModelUpdate.md`;
3. create a detailed `Stage7_Clean_Up.md` plan.

**Do not execute the cleanup/refactor plan in this task.**

Do not redesign the model, change scientific behavior, or launch new long training.

---

# Part 1 — Freeze and synchronize the current validated revision

## 1. Verify current state before doing anything

From the normal primary repository checkout, not from an experimental linked worktree, print and record:

```bash
pwd
git status
git branch --show-current
git rev-parse HEAD
git remote -v
git fetch --all --tags --prune
git log --oneline --decorate -n 15
git worktree list
```

Verify the remote Stage-7 branch:

```bash
git rev-parse origin/perf/pointcloud-smart-cq-stage7
```

Expected reference at task creation:

```text
a03213ae33f14cbffd198c2da65aa8a11e6b2f26
```

If the remote branch has advanced beyond this SHA because additional legitimate Stage-7 documentation/evaluation commits were pushed, inspect those commits and use the latest validated Stage-7 tip only if:

- they do not change the scientific model unexpectedly;
- the Stage-7 final tests/results remain valid;
- the worktree is clean.

Do not silently freeze a different architecture.

## 2. Make the ordinary local checkout canonical

The user's normal repository checkout should end this task:

- on `perf/pointcloud-smart-cq-stage7`;
- fast-forwarded to the exact remote validated tip;
- clean;
- not detached;
- with no uncommitted source changes.

Use fast-forward-only operations.

Do not force-reset over user work.

If the normal checkout contains uncommitted user changes, stop and report instead of overwriting them.

## 3. Consolidate important artifacts out of `.worktrees`

The user does not want important results living only under `.worktrees`.

Audit:

```bash
git worktree list
```

and inspect every linked worktree associated with PointCloud FFM Stages 1–7.

The goal is **not** to delete research history blindly.

For each worktree:

1. identify files that exist only there;
2. classify them as:
   - source already committed elsewhere;
   - canonical evaluation/report artifact;
   - checkpoint/model artifact;
   - temporary logs/cache/runtime files;
3. compare against the normal checkout and current remote branch.

For any canonical Stage-1–7 reports/evaluation outputs needed for reproducibility or `ModelUpdate.md`, ensure a canonical copy exists under the ordinary repository tree, preferably:

```text
0_demo_TurbulentCombustion/_CheckNotes/
0_demo_TurbulentCombustion/figures/generated/
```

For model checkpoints that should not be committed to Git, move/copy the selected important checkpoint(s) into a stable local non-worktree location, for example:

```text
0_demo_TurbulentCombustion/ReleaseArtifacts/GL_rbf_CQ_rc1/
```

and keep that directory ignored if checkpoint size makes Git inappropriate.

Create a manifest:

```text
0_demo_TurbulentCombustion/ReleaseArtifacts/GL_rbf_CQ_rc1/ARTIFACT_MANIFEST.md
```

or, if `ReleaseArtifacts/` is git-ignored, store the manifest in:

```text
0_demo_TurbulentCombustion/_CheckNotes/GL_rbf_CQ_rc1_artifacts.md
```

The manifest should record:

- artifact purpose;
- source worktree/path;
- canonical new path;
- file size;
- SHA256 checksum;
- checkpoint epoch;
- whether EMA-resolved or research checkpoint;
- source Git commit.

### Important

Do not commit multi-GB checkpoints just to freeze the version.

Do not delete any worktree containing unique artifacts until those artifacts have been copied and checksummed.

At the end of this task, it is acceptable to leave linked worktrees registered if deletion is not obviously safe.

The key requirement is:

> no important Stage-7 result or release checkpoint should exist **only** inside `.worktrees`.

## 4. Freeze the source revision

After confirming the source tree and canonical documentation/evidence are committed and pushed, create one annotated release-candidate tag:

```text
gl-rbf-cq-v0.9.0-rc1
```

Tag the exact validated Stage-7 source/documentation commit.

Suggested annotation:

```text
GL_rbf_CQ v0.9.0-rc1 — validated Stage 1–7 PointCloud FFM release candidate
```

Push:

```bash
git push origin perf/pointcloud-smart-cq-stage7
git push origin gl-rbf-cq-v0.9.0-rc1
```

Do not move the tag after it is pushed.

If the tag already exists, verify its SHA rather than overwriting it.

## 5. Record the frozen release candidate

Create:

```text
0_demo_TurbulentCombustion/_CheckNotes/GL_rbf_CQ_RC1_FREEZE.md
```

Keep this concise. Include:

- final tag;
- final commit SHA;
- branch;
- date;
- regression result;
- selected scientific checkpoint;
- selected release-facing model name `GL_rbf_CQ`;
- selected primary config;
- known throughput alternative;
- note that cleanup has not yet been performed.

---

# Part 2 — Create `ModelUpdate.md`

Create:

```text
0_demo_TurbulentCombustion/ModelUpdate.md
```

This should be a polished, self-contained technical document explaining the full evolution from the original PointCloud FFM / `GL_rbf_ENH` implementation through the release-candidate `GL_rbf_CQ`.

It is for:

- the user;
- collaborators;
- future developers;
- paper/rebuttal/reproducibility reference.

Do not write it as raw development notes.

## 1. Table of contents

Start with a clickable Markdown table of contents.

Recommended high-level structure:

```text
1. Executive Summary
2. Original PointCloud FFM / GL_rbf_ENH
3. Stage 1 — Data-Path Bottleneck Closure
4. Stage 2 — Selected Materialization and Diagnostics
5. Stage 3 — Explicit Scaling Characterization
6. Stage 4 — Cached-Streamed Reconstruction
7. Stage 5 — Query-Microbatch RF Training
8. Stage 6 — Compact Query Decoder and Persistent Top-K
9. Stage 7 — Smart CQ / GL_rbf_CQ
10. Accepted vs Rejected Model Changes
11. Final Mathematical Formulation
12. Final Code and Runtime Architecture
13. Staged Quality–Efficiency Comparison
14. Recommended Configurations
15. Checkpoint and Reproducibility Guide
16. Known Limitations / Next Validation
```

Adapt headings if the actual evidence supports a better organization.

## 2. Ground the document in actual Git history and code

Use:

```bash
git log --oneline --decorate --all
git show <stage commits>
git diff <relevant commits>
```

Use the actual Stage 1–7 reports under `_CheckNotes/`.

Do not infer stage behavior only from memory.

For each stage identify:

- relevant commit(s);
- source files changed;
- mathematical behavior changed or preserved;
- runtime/data behavior changed;
- tests/benchmarks;
- accepted/rejected outcome.

## 3. Explain the original model mathematically

Use LaTeX-formatted equations inside Markdown.

At minimum define:

### RF bridge

\[
x_t = (1-t)x_0 + tx_1
\]

with target vector field

\[
v^\star(x_t,t)=x_1-x_0.
\]

Training objective:

\[
\mathcal{L}_{\mathrm{RF}}
=
\mathbb{E}
\left[
\left\|
v_\theta(x_t,t,\mathcal O,\mathbf x_q)
-
(x_1-x_0)
\right\|_2^2
\right].
\]

### Sensor tokens and latent reasoning

Use notation consistent with the actual implementation.

Explain:

- sparse measurements;
- field identity embedding;
- coordinate Fourier encoding;
- latent cross-attention;
- latent self-attention;
- sensor reinjection;
- sensor back-attention / refined sensor tokens.

### Local Top-K RBF

Define the selected neighbor set:

\[
\mathcal N_K(\mathbf x_q)
\]

and a representative RBF weight:

\[
w_{qj}
=
\frac{
\exp\left(-d_{qj}^2/(2\sigma^2)\right)
}{
\sum_{k\in\mathcal N_K(\mathbf x_q)}
\exp\left(-d_{qk}^2/(2\sigma^2)\right)
}.
\]

Explain learned sigma and GLRES sensor-importance adjustment separately and accurately from code.

### Global/local query fusion

Explain F0/ENH query structure and then the compact CQ evolution.

## 4. Stage 1–5: emphasize execution changes versus mathematical changes

For Stages 1–5 clearly state that the core RF / GL-RBF mathematical model was intentionally preserved while execution changed.

Document:

### Stage 1
- legacy versus optimized data path;
- selected GPU transfer;
- scalable CPU index sampling;
- shared mesh coordinates;
- matched fixed-manifest validation.

### Stage 2
- selected-after-full-read normalization;
- append-style diagnostics;
- optimizer zeroing cleanup.

### Stage 3
- measured N_full versus N_query scaling;
- demonstration that query-model computation became dominant.

### Stage 4
- condition context caching;
- query context caching;
- true end-to-end reconstruction streaming;
- Euler/Heun equivalence;
- million-point reconstruction.

Mathematically make clear:

\[
v_\theta
\]

did not change; only the execution schedule changed.

### Stage 5
- effective query count versus execution microbatch;
- one shared RF time and source field across the effective query set;
- weighted microbatch loss;
- gradient equivalence.

Show something like:

\[
\mathcal L
=
\sum_{c=1}^{C}
\frac{N_c}{N}
\mathcal L_c
\]

for query chunks.

## 5. Stage 6: compact-query evolution

Document:

- `GL_rbf_ENH_CQ`;
- 128-D compact query state;
- low-rank query-to-latent readout;
- additive fusion;
- query memory/runtime benefit;
- CQ-LR-128 quality penalty;
- CQ-Full;
- rescue160;
- CQ-Balanced 192/224 structured-concat attempt;
- why CQ-Balanced was rejected before training.

Make the accepted/rejected distinction explicit.

### Persistent geometry-only Top-K

Explain:

\[
\{\mathrm{idx}_{qk},d^2_{qk}\}
\]

depends only on query geometry / sensor geometry / mask / K and may be cached independently of sensor values.

Show the separation between:

```text
persistent geometry
per-condition sensor/latent state
per-NFE dynamic state
```

## 6. Stage 7: final `GL_rbf_CQ`

Use `GL_rbf_CQ` as the public name.

Explain that the final model retains:

```text
latent_dim = 256
num_latents = 128
num_latent_blocks = 4
cq_query_dim = 128
readout rank = 64
K = 32
```

### Sinusoidal time embedding

Use an equation consistent with code, e.g.

\[
\gamma(t)
=
[
\sin(\omega_1 t),\cos(\omega_1 t),\dots,
\sin(\omega_m t),\cos(\omega_m t)
].
\]

Then show FiLM:

\[
\tilde h_q
=
h_q\odot(1+s(t))+b(t).
\]

Note the zero-initialized FiLM projection and why the static condition cache remains time-independent.

### Explicit measurement/support features

For field \(f\):

\[
s_f(\mathbf x_q)
=
\sum_{j\in\mathcal N_K(\mathbf x_q)}
w_{qj}
\mathbb{1}[f_j=f],
\]

\[
\hat y_f(\mathbf x_q)
=
\frac{
\sum_j
w_{qj}\,
\mathbb{1}[f_j=f]\,y_j
}{
s_f(\mathbf x_q)+\varepsilon
}.
\]

Explain that these features reuse the same Top-K geometry and do not add another KNN search.

### EMA

Explain parameter EMA:

\[
\bar\theta_k
=
\beta\bar\theta_{k-1}
+
(1-\beta)\theta_k,
\qquad
\beta=0.999.
\]

Very clearly document the final corrected semantics:

- average trainable parameters only;
- copy frozen parameters/buffers exactly;
- use live frozen state when repairing old Stage-7 checkpoints.

### Stronger condition core

Explain why increasing latent width from 128 to 256 raises once-per-condition capacity without widening the repeated 128-D query path.

## 7. Final model equation / information flow

Create one compact final mathematical schematic.

Something conceptually like:

\[
\mathcal O
\rightarrow
Z
\rightarrow
\{g,\tilde H_s\}
\]

for sparse-condition encoding,

\[
\mathcal N_K(\mathbf x_q)
\rightarrow
\{l_q,\hat y_q,s_q\}
\]

for local learned and explicit features,

\[
h_q(x_t,t,\mathbf x_q)
\rightarrow
\tilde h_q
\]

for dynamic query + FiLM,

and

\[
v_\theta
=
v_{\mathrm{coarse}}
+
H_\theta(
\tilde h_q,
g_q,
l_q,
r_q,
\hat y_q,
s_q
)
\]

for the final compact velocity.

Use actual code to ensure notation matches implementation.

## 8. Accepted vs rejected table

Include a table similar to:

| Change | Outcome | Retained in GL_rbf_CQ? | Reason |
|---|---|---|---|
| optimized data path | accepted | yes | lower CPU/RSS/step cost |
| cached streaming | accepted | yes | million-point inference |
| query microbatching | accepted | yes | exact-gradient memory control |
| persistent Top-K | accepted | yes | repeated inference |
| CQ-LR 128 | partially accepted | base of final query decoder | efficient but latent128 quality penalty |
| wider CQ-Balanced concat | rejected | no | restored too much F0 cost |
| explicit SDPA rewrite | rejected | no | slower than MHA |
| fused AdamW | optional | no default | only ~1.8% gain |
| latent256 condition core | accepted | yes | major quality recovery |
| EMA | accepted | yes | stable final evaluation |
| time FiLM | accepted | yes | smarter dynamic conditioning |
| measurement/support shortcut | accepted | yes | explicit local measurement info |

Base every row on actual evidence.

## 9. Staged evaluation / visualization

Create a structured set of figures, not one overloaded plot.

Use existing checkpoints and artifacts wherever possible.

Do not launch new long training.

New short deterministic/fixed-manifest evaluations are allowed when needed to make comparisons consistent.

### Figure group A — Model evolution / architecture

Create a clean schematic showing:

```text
GL_rbf_ENH
  -> execution scaling (Stages 1–5)
  -> compact CQ (Stage 6)
  -> GL_rbf_CQ (Stage 7)
```

Include rejected branches visually but clearly marked as rejected:

```text
CQ-Balanced 192/224 -> rejected
explicit SDPA -> rejected
```

### Figure group B — Quality convergence

Use matched fixed-manifest values for available checkpoints.

At minimum compare:

```text
F0 / GL_rbf_ENH
CQ-LR-128
CQ-LR-256 (mark incomplete if applicable)
GL_rbf_CQ
```

Use exact checkpoint epochs and label incomplete runs.

### Figure group C — Reconstruction quality

Use one or more matched snapshots with the same:

- observation fields;
- sensor count/layout;
- RF seed;
- solver;
- NFE.

Do not compare unmatched sensor layouts.

If practical, evaluate more than the single historical snapshot, for example a small fixed set of 3–5 validation snapshots, without retraining.

Produce:

- field-average relative L2;
- worst-field U1;
- representative field plates.

### Figure group D — Efficiency evolution

Include:

- training step time;
- peak allocated memory;
- 1M-query NFE4 persistent inference;
- geometry build/cache cost where relevant.

Use same-hardware matched numbers.

### Figure group E — Stage 1–5 execution changes

Do not pretend Stages 1–5 represent different learned models.

Use execution figures/tables showing:

- data path speed/RSS;
- model-vs-data scaling;
- legacy vs streamed reconstruction;
- microbatch memory scaling;
- persistent Top-K reuse.

## 10. Checkpoint inventory

Create a section listing important model checkpoints available locally.

For each:

```text
model label
run
epoch
path
source commit
architecture
EMA/live
normalization stats
RF-prior checksum if available
status:
    formal
    partial
    rejected
    diagnostic
```

If a stage has no meaningful trained checkpoint because it was an execution-only revision, say so.

Do not manufacture a “Stage 2 model checkpoint” when the mathematics was unchanged.

## 11. Final recommended configurations

Use release-facing names:

### `GL_rbf_CQ` — recommended balanced/default

Document exact config:

```yaml
backbone: GL_rbf_ENH_CQ  # frozen internal identifier; public name GL_rbf_CQ
latent_dim: 256
num_latents: 128
num_latent_blocks: 4
cq_query_dim: 128
cq_readout_mode: lowrank
cq_readout_rank: 64
cq_readout_heads: 4
cq_fusion_mode: additive
cq_time_conditioning: sinusoidal_film
cq_measurement_support_mode: rbf_value_support
model_ema_enabled: true
model_ema_decay: 0.999
model_ema_eval: true
gather_mode: topk_rbf_glres
gather_topk: 32
```

### `GL_rbf_CQ-fast` — throughput option

Use frozen CQ-LR-128 settings and persistent Top-K.

### `GL_rbf_ENH` — legacy/reference

Retain only for reproducibility/comparison.

## 12. Final limitations

Be clear that:

- the decisive release-candidate evidence is from the current 2-D turbulent-combustion case;
- final v1.0 requires formal large 3-D wind-farm/geothermal validation;
- Senseiver and Latent FM remain structurally different references;
- CQ-LR-256 is incomplete if it was not completed to 1000 epochs.

---

# Part 3 — Create `Stage7_Clean_Up.md`

Create:

```text
0_demo_TurbulentCombustion/Stage7_Clean_Up.md
```

This is a **plan only**.

Do not execute any cleanup, deletion, source relocation, or public-name refactor in this task.

The document should be sufficiently detailed that a later Codex run can execute it safely.

## 1. Goal

The cleanup goal is:

> Turn the research-heavy Stage-1–7 PointCloud FFM repository into a clean next-development baseline centered on `GL_rbf_CQ`, while preserving frozen reproducibility of all earlier work.

The cleaned code should be easy for collaborators to:

- understand;
- train;
- evaluate;
- run on new 3-D datasets;
- use persistent geometry caching;
- choose balanced versus throughput presets.

## 2. Audit the current repository

Before writing the plan, inspect:

```text
src/
tests/
Save_config/
_CheckNotes/
figures/
Save_TrainedModel/
worktrees
legacy scripts
benchmark scripts
evaluation scripts
```

Identify:

- runtime-critical files;
- research-only files;
- duplicate functionality;
- obsolete legacy paths;
- rejected-model code;
- compatibility code still needed for old checkpoints;
- absolute local paths;
- generated files accidentally tracked;
- scripts that should move under research/archive;
- files imported by production runtime.

Do not remove anything now.

## 3. Proposed clean directory structure

Propose a practical final structure, for example:

```text
0_demo_TurbulentCombustion/
├── README.md
├── ModelUpdate.md
├── configs/
│   ├── gl_rbf_cq.yaml
│   ├── gl_rbf_cq_fast.yaml
│   └── legacy_gl_rbf_enh.yaml
├── src/
│   ├── models/
│   │   ├── gl_rbf_cq.py
│   │   ├── gl_rbf_core.py
│   │   └── priors.py
│   ├── data/
│   │   └── pointcloud_data.py
│   ├── training/
│   │   ├── pointcloud_ffm_trainer.py
│   │   └── ema.py
│   ├── inference/
│   │   ├── reconstruction.py
│   │   └── geometry_cache.py
│   └── utils/
├── examples/
│   ├── train_gl_rbf_cq.sh
│   ├── reconstruct_gl_rbf_cq.py
│   └── repeated_inference_cached_geometry.py
├── tests/
└── research_archive/
```

Do not force this exact structure if the current code dependencies suggest a safer one.

## 4. Public naming plan

The release-facing name should become:

```text
GL_rbf_CQ
```

Plan a backward-compatible rename:

- allow config `backbone: GL_rbf_CQ`;
- internally map it to the cleaned CQ implementation;
- retain loading support for old metadata with:
  `GL_rbf_ENH_CQ`;
- preserve old class alias if needed.

Do not break old Stage-6/7 checkpoints.

The public documentation should stop using:

```text
Stage7-All256
S7-B
```

except in historical sections.

## 5. What should be retained

Plan to retain production support for:

- GL_rbf_CQ balanced model;
- GL_rbf_CQ-fast throughput preset;
- persistent Top-K geometry cache;
- cached-streamed inference;
- query microbatch training;
- optimized data path;
- EMA;
- fixed-manifest evaluation utilities that are genuinely useful;
- F0/GL_rbf_ENH checkpoint compatibility if practical;
- core tests.

## 6. What should be archived or removed from the active path

Plan how to archive—not blindly delete—things such as:

- CQ-Balanced 192/224 experiment code paths if no longer needed;
- old one-off benchmark launchers;
- Stage-specific temporary analyzers;
- obsolete generated figures;
- redundant run configs;
- legacy data-path A/B machinery no longer needed by collaborators;
- duplicate evaluation scripts;
- old worktree-only runtime logs;
- stale absolute-path configs;
- temporary PID/env/monitor outputs.

Separate:

```text
remove from active runtime
archive for research history
retain for compatibility
```

## 7. Simplify `Model.py`

The current `Model.py` has accumulated many model families and experiment variants.

Plan how to extract or isolate:

- GL-RBF core;
- GL_rbf_CQ;
- latent readout;
- priors;
- persistent geometry helpers.

The plan must minimize risk.

Do not propose a giant one-commit rewrite.

Recommend safe extraction order with numerical-equivalence tests after each extraction.

## 8. Simplify training entry point

Plan how to make collaborator-facing training simple.

A user should ideally run something like:

```bash
python src/train_pointcloud_ffm.py \
  --config configs/gl_rbf_cq.yaml
```

or an equivalent clean launcher.

Remove the need to understand Stage1–7 historical flags.

Keep advanced overrides available but not dominant.

## 9. Clean config presets

Propose three presets:

### `gl_rbf_cq.yaml`

Balanced/recommended model.

### `gl_rbf_cq_fast.yaml`

CQ-LR-128 throughput model.

### `legacy_gl_rbf_enh.yaml`

Historical F0 reference.

All paths should be relative or CLI-overridable.

No `/home/wanglz/...` paths in release configs.

## 10. Exported release checkpoint

Plan a coworker-safe checkpoint export.

The release checkpoint should contain a directly loadable resolved model state:

```text
EMA trainable parameters
+
exact live frozen parameters/buffers
```

so coworkers do not need to understand EMA repair semantics.

Recommended release artifact name:

```text
GL_rbf_CQ_v0.9.0-rc1_e1000.pt
```

Include metadata:

- public model name;
- internal backbone identifier;
- architecture config;
- field names;
- normalization stats/reference;
- source Git tag/SHA;
- epoch;
- RF-prior checksum;
- training dataset identifier;
- EMA provenance.

Keep original research checkpoint separately.

## 11. Quick-start documentation

Plan a clean `README.md` workflow:

1. environment setup;
2. dataset format;
3. train;
4. resume;
5. reconstruct once;
6. persistent-geometry repeated reconstruction;
7. 1M-point example;
8. selecting balanced vs fast preset.

Do not require collaborators to read `_CheckNotes/`.

## 12. Release tests

Define the cleanup acceptance suite:

### Unit/regression
- existing maintained tests;
- checkpoint compatibility;
- EMA export/load;
- geometry cache;
- microbatch equivalence.

### Fresh-clone smoke
- environment imports;
- tiny CPU test;
- GPU forward/backward;
- tiny training;
- checkpoint save/reload;
- reconstruction.

### Numerical equivalence
Compare pre-cleanup frozen `gl-rbf-cq-v0.9.0-rc1` against cleaned code on:

- fixed deterministic small tensors;
- one fixed-manifest batch;
- one matched reconstruction.

### Performance sanity
Ensure cleanup does not materially regress:

- B128/Q4096 step;
- 1M/NFE4 persistent inference.

## 13. CI plan

Recommend a minimal GitHub Actions workflow for:

- syntax/import;
- fast CPU tests;
- config parsing;
- small checkpoint-free smoke tests.

Do not require GPU/KeOps benchmarking in hosted CI.

## 14. 3-D readiness

The cleanup plan should ensure the code makes as few 2-D assumptions as possible.

Audit:

- `coord_dim`;
- fixed-grid assumptions;
- visualization-only 2-D logic;
- HDF5 layout assumptions;
- mesh-fixed vs variable mesh;
- persistent geometry behavior;
- dataset adapters.

The release candidate may remain v0.9 until formal 3-D validation is complete.

## 15. Cleanup execution phases

Provide a small number of safe phases for the later task, e.g.:

```text
A. archive/reorganize research artifacts
B. isolate GL_rbf_CQ runtime modules
C. simplify configs/examples/public naming
D. export release checkpoint + fresh-clone validation
```

Each phase must have a test gate.

Again:

> DO NOT EXECUTE THESE PHASES IN THIS TASK.

---

# Final task report

At the end of this task, report only:

1. synchronized branch + SHA;
2. frozen `gl-rbf-cq-v0.9.0-rc1` tag + SHA;
3. worktree/artifact consolidation summary;
4. path to `ModelUpdate.md`;
5. figures/evaluations newly produced for `ModelUpdate.md`;
6. path to `Stage7_Clean_Up.md`;
7. confirmation that the cleanup/refactor plan was **not executed**;
8. any unresolved artifact/checkpoint-location issues.

Do not start the cleanup implementation.
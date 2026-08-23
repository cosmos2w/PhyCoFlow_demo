# Stage 7 Goal — Smart CQ: Stronger Condition Representation + Smarter Training

## Repository / frozen reference

Repository:

`cosmos2w/PhyCoFlow_demo`

Verified current remote reference:

`6ac549b26a229222a209552bac344baeb86b7a4e`

Both current performance branches point to this commit:

- `perf/pointcloud-ffm-query-decoder-stage6`
- `perf/pointcloud-ffm-field-reconstruction`

This commit contains the validated:

- `GL_rbf_ENH_CQ`;
- CQ-LR compact query decoder;
- Stage 1–5 execution optimizations;
- persistent geometry-only Top-K cache;
- CQ-Balanced negative-result evidence;
- current clean latent-width comparison artifacts.

Before Stage-7 source edits, create an immutable annotated tag:

```text
gl-rbf-enh-cq-v1
```

at exactly:

```text
6ac549b26a229222a209552bac344baeb86b7a4e
```

Do not move existing branches or merge to `main`.

For Stage 7, create/use one development branch from this exact commit, e.g.:

```text
perf/pointcloud-ffm-stage7-smart-cq
```

Keep the branch-management work minimal.

---

# 1. Stage-7 scientific objective

Do **not** widen the CQ query decoder again.

The previous CQ-Balanced experiment showed that restoring wider per-query capacity and F0-like structured fusion recovered too much of F0's computational cost before scientific training was even justified.

Stage 7 follows a different principle:

> Keep the CQ-LR query decoder cheap, but make the condition representation and RF training smarter.

The Stage-7 default query decoder remains:

```yaml
backbone: "GL_rbf_ENH_CQ"
cq_query_dim: 128
cq_readout_mode: "lowrank"
cq_readout_rank: 64
cq_readout_heads: 4
cq_fusion_mode: "additive"
```

Persistent Top-K remains unchanged.

Implement four **orthogonal, optional** improvements in one source revision:

1. model EMA for training/evaluation;
2. stronger sensor/global latent core while keeping the 128-D CQ query decoder;
3. sinusoidal timestep embedding + FiLM-style modulation of the dynamic query branch;
4. explicit raw measurement + support features computed from the existing Top-K geometry.

All new behaviors must be switchable from YAML and default to the historical CQ behavior when disabled.

Do not implement them as four separate code branches.

---

# 2. Why these four changes

## EMA

Latent Flow Matching maintains an EMA velocity model and evaluates/samples with EMA weights.

This is nearly zero-risk:

- no inference architecture change;
- no query-cost increase;
- tiny training overhead;
- potentially smoother/better generalization.

## Stronger condition core

Senseiver invests much more capacity in sensor-to-latent/global reasoning than current CQ, while the expensive query path remains conceptually simple.

For million-point inference, condition-core computation occurs once per condition while query-decoder computation repeats for every point/NFE.

Therefore Stage 7 should test stronger global reasoning **without increasing `cq_query_dim`**.

Primary stronger-core setting:

```yaml
latent_dim: 256
num_latents: 128
num_latent_blocks: 4
```

Do not simultaneously increase latent blocks/count in the primary Stage-7 runs.

The current clean latent-256 experiments already in progress/completed should be reused as reference evidence; do not launch a duplicate latent-256-only run.

## Time FiLM

Current CQ largely represents time through scalar `t` concatenated to each point-state token.

Latent FM uses sinusoidal timestep embedding and FiLM/AdaGN-style modulation.

Stage 7 should add a cheap global per-sample time embedding and use it only to modulate the dynamic CQ point branch.

Do **not** make the sensor/latent condition context time-dependent, because that would destroy the current once-per-condition/NFE caching advantage.

## Explicit measurement/support shortcut

CQ's local branch currently passes learned globally refined sensor features through RBF gather.

This is expressive, but the query head has no direct low-dimensional channel saying:

- what raw values were physically measured nearby;
- which fields actually have local support;
- how much RBF weight/support belongs to each field.

Use the already-computed Top-K indices/distances to expose this information without another KNN search.

---

# 3. Hard invariants

Stage 7 must preserve:

- CQ-LR query width = 128 in all primary runs;
- rank-64 / four-head low-rank latent readout;
- additive CQ fusion;
- K=32;
- KeOps neighbor semantics;
- learnable RBF sigma;
- GLRES sensor-importance mechanism;
- field embeddings;
- Fourier coordinate encoding;
- latent sensor re-injection logic;
- sensor back-attention/refinement;
- RFF prior;
- RF bridge / target `x1 - x0`;
- RF loss definition;
- query count = 4096;
- Stage 1–5 optimized data path;
- cached-streamed reconstruction;
- persistent geometry-only Top-K;
- Euler/Heun semantics;
- observation-consistency behavior;
- old F0 and CQ checkpoint compatibility.

Do not add:

- another query-width sweep;
- another CQ fusion family;
- anchors;
- new latent spatial fields;
- neural operators;
- autoencoder/latent FM machinery;
- larger effective query counts.

---

# 4. Config surface

Add a clearly grouped Stage-7 section.

Recommended keys:

```yaml
# ==========================================================
# Stage 7 — Smart CQ options
# ==========================================================

# ----- model EMA -----
model_ema_enabled: false
model_ema_decay: 0.999
model_ema_eval: true

# ----- CQ time conditioning -----
cq_time_conditioning: "scalar_concat"
# ["scalar_concat", "sinusoidal_film"]

cq_time_embed_dim: 128
cq_time_max_period: 10000.0
cq_time_film_zero_init: true

# ----- explicit measurement/support shortcut -----
cq_measurement_support_mode: "none"
# ["none", "rbf_value_support"]

cq_measurement_support_normalize: true

# stronger condition core continues to use existing architecture keys:
latent_dim: 128
num_latents: 128
num_latent_blocks: 4
```

All defaults must reproduce the frozen CQ-v1 behavior.

Do not add a redundant `strong_core_enabled` boolean; use the existing `latent_dim` setting.

Checkpoint metadata must record all new Stage-7 options.

---

# 5. EMA implementation

Add a lightweight EMA utility for `PointCloudFFM`, similar in semantics to the existing latent-FM EMA but independent of the baseline code.

## Requirements

When:

```yaml
model_ema_enabled: false
```

training/evaluation behavior must remain unchanged.

When enabled:

1. initialize shadow weights from current trainable model parameters;
2. update EMA after each successful `optimizer.step()`;
3. default decay:
   `0.999`;
4. save EMA state in checkpoints;
5. reload EMA state on resume;
6. use EMA weights for validation/reconstruction when:
   `model_ema_eval: true`;
7. restore live training weights after evaluation;
8. save both live model state and EMA state.

Do not permanently overwrite live parameters.

Use a context manager similar to:

```python
with model_ema.evaluation_weights(model):
    ...
```

or equivalent.

## Best checkpoint semantics

If EMA evaluation is enabled:

- validation loss used for best-checkpoint selection should be measured using EMA weights;
- checkpoint must still contain live weights + EMA shadow;
- record `best_selection_weights: "ema"` in metadata.

Training loss remains computed with live weights.

## Correctness tests

Test:

- disabled EMA leaves old training path unchanged;
- EMA update formula;
- evaluation applies/restores weights exactly;
- checkpoint round trip;
- resume round trip;
- no parameter left overwritten after validation.

---

# 6. Stronger sensor/global latent core

Do not change CQ query width.

The Stage-7 strong-core setting is:

```yaml
latent_dim: 256
num_latents: 128
num_latent_blocks: 4
cq_query_dim: 128
cq_readout_rank: 64
```

The existing CQ low-rank readout must work with latent widths 128 and 256.

`CompactLatentReadout` should continue to project:

```text
latent_dim -> K rank 64
latent_dim -> V query_dim 128
```

so per-query attention width remains compact.

The stronger core may increase:

- sensor token width;
- latent attention cost;
- condition-context build time;

but should not cause query hidden states to become 256-D.

Add/retain component timing to report:

```text
condition_context_ms
query_chunk_ms
```

separately.

Do not widen `cq_query_dim` in Stage 7.

---

# 7. Sinusoidal timestep embedding + FiLM

Add an optional CQ-only module.

Recommended conceptual implementation:

```python
class CQTimeEmbedding(nn.Module):
    # sinusoidal scalar t embedding
    # -> Linear
    # -> SiLU
    # -> Linear
```

Use standard diffusion/flow sinusoidal frequencies with configurable:

```yaml
cq_time_embed_dim: 128
cq_time_max_period: 10000.0
```

## Important caching constraint

Do not inject timestep information into:

- sensor tokens;
- latent condition encoding;
- refined sensor features;
- persistent geometry;
- coordinate query readout cache.

These are intentionally static over the ODE trajectory.

Time conditioning should affect only the **dynamic point-state branch**.

## Recommended modulation

Preserve the existing scalar `t` input to the point encoder for maximum backward conceptual compatibility.

After:

```python
point_q = cq_point_encoder([coord_feat, x_t, scalar_t])
```

apply:

```python
time_emb = cq_time_embed(t)
scale, shift = cq_time_film(time_emb).chunk(2, dim=-1)

point_q = cq_time_norm(point_q)
point_q = point_q * (1 + scale[:, None, :]) + shift[:, None, :]
```

or an equivalent residual FiLM formulation.

Prefer:

- `LayerNorm(cq_query_dim)` or another simple normalization;
- final FiLM projection zero-initialized if:
  `cq_time_film_zero_init: true`.

At initialization, zero-init should make the FiLM path close to identity.

Do not make the low-rank latent readout time-dependent.

## Config behavior

```yaml
cq_time_conditioning: "scalar_concat"
```

must instantiate/use no Stage-7 FiLM behavior.

```yaml
cq_time_conditioning: "sinusoidal_film"
```

enables the new modules.

Existing CQ checkpoints without these keys must still strict-load under the historical mode.

---

# 8. Explicit raw measurement + support shortcut

Add:

```yaml
cq_measurement_support_mode: "none"
# or
cq_measurement_support_mode: "rbf_value_support"
```

This must use the **same Top-K neighbor search** already needed for local RBF conditioning.

No second KNN search is allowed.

## Feature definition

For each query and field `f`, using existing Top-K geometry:

```text
neighbor indices
neighbor d²
valid mask
raw obs_values
obs_field_ids
```

compute a pure geometry/RBF weighting.

Recommended:

```python
base_logits = -topk_d2 / (2 * sigma**2 + eps)
base_weights = softmax(masked(base_logits))
```

Do not include the learned sensor-importance bias in the explicit raw-measurement statistics.

For field `f`:

```text
support_f =
    sum_k base_weight_k * I(field_k == f)

value_f =
    sum_k base_weight_k * I(field_k == f) * obs_value_k
    / (support_f + eps)
```

If no support exists:

```text
support_f = 0
value_f = 0
```

The explicit feature is:

```text
[value_1 ... value_F,
 support_1 ... support_F]
```

For five fields this is only 10 scalar features/query.

The feature uses normalized observation values, consistent with the current dataset/model inputs.

## Query-head integration

Keep it explicit and cheap.

Recommended:

```python
measurement_support = [B,N,2F]

if cq_measurement_support_normalize:
    measurement_support = LayerNorm(2F)(measurement_support)

head_input = concat(
    fused_cq_128,
    measurement_support_2F,
)
```

Then use a Stage-7 head only when this feature is enabled:

```text
(128 + 2F) -> 128 -> 128 -> fields
```

For F=5:

```text
138 -> 128 -> 128 -> 5
```

This adds negligible per-query width compared with the failed 192/224-D CQ-Balanced designs.

When disabled, preserve the exact historical CQ head and state-dict shape.

## One-search requirement during training

Do not implement this by:

1. computing learned `local_cond` using Top-K;
2. calling Top-K again for measurement/support.

Instead refactor/add a CQ helper that can derive both from the same neighbor result:

```text
one KNN
→ learned local condition
→ explicit measurement/support
```

For persistent/cache execution:

```text
precomputed geometry
→ learned local condition
→ explicit measurement/support
```

with zero new KNN.

## Context data

When the feature is enabled, the condition context may additionally retain:

```text
obs_values
obs_field_ids
```

These are condition-static and small.

Do not put them into the persistent geometry cache itself.

## Static caching

For `reconstruction_cache_level="static_features"`:

cache:

- learned local condition;
- CQ coordinate latent readout;
- measurement/support feature.

For `geometry`:

use cached geometry but recompute the cheap static statistics as current semantics dictate.

---

# 9. Interaction of FiLM and measurement/support

The Stage-7 full query path should remain conceptually:

```text
dynamic:
    coord + x_t + t
        -> CQ point encoder
        -> optional sinusoidal time FiLM
        -> point_q 128

static condition:
    sensors
        -> stronger/global latent core
        -> global_q
        -> low-rank query readout

local:
    one Top-K geometry
        -> learned refined-sensor RBF local_cond
        -> raw measurement/support shortcut

fusion:
    historical CQ additive fusion of
        point_q
        global_q
        local_q
        low-rank query_global

head:
    historical CQ head
    OR
    [fused_128, explicit_measurement_support] -> compact Stage-7 head
```

Do not change CQ additive fusion itself in Stage 7.

---

# 10. Stage-7 correctness gates

Before any new scientific training, run the full regression suite.

Add focused tests for all new options.

## A. Backward compatibility

Frozen CQ-v1 configuration with all Stage-7 options disabled must:

- strict-load an existing CQ-LR checkpoint;
- produce the same output as before;
- preserve old checkpoint metadata interpretation;
- preserve persistent Top-K behavior.

## B. EMA

Test update/apply/restore/checkpoint/resume.

## C. Time FiLM

Test:

- disabled path exact old behavior;
- enabled output shape;
- gradients reach time embedding/FiLM;
- zero-init starts close to historical point branch;
- cached-streamed vs normal execution equivalence;
- FiLM does not cause condition context to be rebuilt per NFE.

## D. Measurement/support

Test:

- no second KNN;
- per-field value/support values on a hand-constructed toy example;
- missing-field support=0/value=0;
- gradients reach learnable RBF sigma through support/value path if intended;
- fresh geometry vs persistent geometry match;
- changing obs_values changes raw value feature while reusing geometry;
- changing field IDs changes fieldwise statistics;
- zero post-build KNN calls with persistent geometry.

## E. Latent widths

Test CQ-LR at:

```text
latent_dim=128
latent_dim=256
```

with both Stage-7 features enabled.

## F. Microbatch training

Monolithic and query-microbatched execution must remain equivalent for:

```text
Stage7 options off
Stage7 all-on
```

Compare:

- loss;
- all gradients;
- RBF sigma gradient;
- one optimizer update.

---

# 11. Pre-training efficiency benchmark

Do not repeat the CQ-Balanced mistake of spending long training before measuring cost.

Benchmark:

```text
Frozen CQ-LR-128 / latent128
Stage7-Cond128
Stage7-All256
```

where:

## Frozen reference

```yaml
latent_dim: 128
model_ema_enabled: false
cq_time_conditioning: "scalar_concat"
cq_measurement_support_mode: "none"
```

## Stage7-Cond128

```yaml
latent_dim: 128
model_ema_enabled: true
model_ema_decay: 0.999
cq_time_conditioning: "sinusoidal_film"
cq_measurement_support_mode: "rbf_value_support"
```

## Stage7-All256

```yaml
latent_dim: 256
model_ema_enabled: true
model_ema_decay: 0.999
cq_time_conditioning: "sinusoidal_film"
cq_measurement_support_mode: "rbf_value_support"
```

Do not launch an isolated EMA-only, FiLM-only, or measurement-only long run.

Measure at:

```text
B128 / Q4096
```

and model-only:

```text
Q = 4k, 16k, 65k
```

Report:

```text
condition_context_ms
query_forward_ms
backward_ms
optimizer_ms
full_step_ms
peak allocated/reserved
```

Also benchmark repeated:

```text
1M queries
Euler NFE=4
persistent geometry + static_features
```

EMA is not part of inference cost; benchmark EMA weights as ordinary weights.

---

# 12. Efficiency acceptance before formal screens

Stage7-Cond128 should retain most CQ-LR efficiency.

Desirable:

```text
full step <= 1.10 × CQ-LR step time
```

Stage7-All256 may cost more fixed condition processing, but should remain materially cheaper than F0.

Required for scientific screening:

```text
full training step >= 1.10x faster than F0
persistent 1M/NFE4 >= 1.15x faster than F0
```

Do not require Stage7-All256 to be as fast as CQ-LR.

If All256 fails those minimums badly, still retain the code/config option but do not promote it to a formal long run.

---

# 13. Reuse existing latent-256 evidence

The current repository/PR notes that clean latent-256 runs are already in progress or may finish while Stage-7 code is being developed.

Do not duplicate a latent-256-only CQ run.

If available, ingest its results as:

```text
CQ-LR latent256, no EMA, no FiLM, no measurement shortcut
```

This provides a useful existing reference for the effect of stronger condition capacity.

Do not modify those run artifacts.

---

# 14. Formal Stage-7 experiment matrix — combined, not one-by-one

Time is limited.

Do not run a 2^4 factorial study.

Use the existing frozen references plus **two new Stage-7 configurations**.

## Existing references — no new training

Use:

1. clean CQ-LR latent128;
2. clean F0-ENH;
3. clean latent256 CQ-LR run, if completed.

## New Run S7-A — Smart conditioning at latent128

```yaml
backbone: "GL_rbf_ENH_CQ"

latent_dim: 128
num_latents: 128
num_latent_blocks: 4

cq_query_dim: 128
cq_readout_mode: "lowrank"
cq_readout_rank: 64
cq_readout_heads: 4
cq_fusion_mode: "additive"

model_ema_enabled: true
model_ema_decay: 0.999
model_ema_eval: true

cq_time_conditioning: "sinusoidal_film"
cq_time_embed_dim: 128

cq_measurement_support_mode: "rbf_value_support"
cq_measurement_support_normalize: true
```

This tests the combined benefit of:

```text
EMA + better time conditioning + explicit raw measurement/support
```

without changing global latent width.

## New Run S7-B — Full Stage-7 stack

Same as S7-A, except:

```yaml
latent_dim: 256
```

This tests:

```text
stronger condition core
+
EMA
+
time FiLM
+
explicit measurement/support
```

Do not add other architecture differences.

---

# 15. Training protocol

Use the same clean protocol used for the current CQ/F0 comparisons unless a dataset-specific formal config explicitly differs.

For the turbulent-combustion reference:

```text
seed = 42
batch = 128
n_query_points = 4096
monolithic queries
same observations
same optimized data path
same RFF prior
same RF objective
same optimizer family/hyperparameters
```

Do not increase query supervision.

Use:

```text
scheduler horizon = 1000
```

Run both S7-A and S7-B to **200 epochs first**, preferably concurrently on separate idle GPUs.

Do not run them serially one-by-one.

Save/check:

```text
1, 20, 40, 60, 100, 150, 200
```

Use fixed-manifest evaluation.

Validation/reconstruction should use EMA weights for Stage-7 runs.

Training curves remain based on live weights.

---

# 16. 200-epoch joint decision

At epoch 200 compare:

```text
F0
CQ-LR-128
existing CQ-LR-256 if available
S7-A
S7-B
```

Primary questions:

1. Does S7-A move quality upward without sacrificing CQ efficiency?
2. Does stronger latent capacity in S7-B add a further useful quality gain?
3. Does S7-B's fixed condition cost remain acceptable for 3-D scaling?

## Continue to 1000

Continue at most **one primary winner** to 1000 epochs unless both are nearly tied and GPU time is abundant.

Preferred selection score:

1. controlled fixed-manifest RF;
2. matched reconstruction;
3. worst important field;
4. training step;
5. persistent 1M NFE4 inference.

A run is worth continuing if by epoch 200 it:

- clearly improves on CQ-LR-128 quality;
- has recovered a meaningful fraction of the F0 quality gap;
- does not introduce a serious field-specific degradation;
- satisfies the minimum efficiency gate.

If S7-A and S7-B are both weak, stop rather than opening more architectural variants.

---

# 17. Formal success criteria

A Stage-7 model is a strong candidate to replace CQ-LR if after the long run:

```text
best validation gap vs F0 <= 3%
controlled fixed-manifest RF gap <= 3%
mean matched reconstruction gap <= 3–4%
no important field >5% worse
```

while retaining at least:

```text
>= 1.10x train-step speedup vs F0
>= 1.15x repeated 1M/NFE4 inference speedup vs F0
```

A model within ~1–2% of F0 quality with those efficiency gains should become the preferred ENH-CQ configuration.

If quality remains ~5–10% behind F0:

- keep F0 as quality model;
- retain the best Stage-7/CQ-LR as throughput model;
- do not widen the query decoder again.

---

# 18. Compare to Senseiver and latent FM after Stage-7 runs

Do not modify the baseline models.

Use their existing trainers/checkpoints/configs as references:

```text
src/train_Det_Baseline.py   -> Senseiver
src/train_Gen_Baseline.py   -> Latent FM
```

For the selected Stage-7 candidate, report a compact matched comparison:

```text
model parameters
training step / epoch time
peak memory
inference end-to-end
network evaluations / NFE
reconstruction metrics
```

Be explicit that:

- Senseiver is deterministic one-pass supervised reconstruction;
- CQ is full-function-space generative RF;
- latent FM evolves a learned compressed latent field.

Do not interpret structurally different inference costs as pure implementation failures.

---

# 19. Preserve persistent Top-K

Persistent geometry-only Top-K is complete and validated.

Stage 7 must not rewrite it.

For repeated inference:

```text
build geometry once
cache_level = static_features
NFE = 4
```

Measurement/support must reuse the same geometry.

At 1M queries report separately:

```text
geometry build time
geometry memory
static context/cache memory
steady NFE2
steady NFE4
```

---

# 20. Output/evidence package

Create:

```text
_CheckNotes/Stage7_smart_cq/
```

Suggested contents:

```text
README.md
implementation/
benchmarks/
configs/
screen_200/
formal_selected/
comparison/
```

Produce:

1. config table;
2. correctness/equivalence results;
3. parameter counts;
4. component timing;
5. 200-epoch convergence comparison;
6. fixed-manifest comparison;
7. matched reconstruction;
8. 1M persistent-inference timing;
9. quality-efficiency Pareto versus F0/CQ-LR;
10. compact Senseiver/latent-FM comparison.

---

# 21. Final deliverables

Report:

1. frozen tag and SHA;
2. Stage-7 branch/commit SHA;
3. files changed;
4. new config keys;
5. exact backward-compatibility result;
6. full regression count;
7. EMA correctness;
8. FiLM correctness/caching behavior;
9. measurement/support toy-value correctness;
10. proof of no additional KNN search;
11. latent128/256 model summaries;
12. pre-training cost benchmark;
13. S7-A and S7-B 200-epoch results;
14. selected continuation decision;
15. long-run result if continued;
16. comparison against F0, CQ-LR, Senseiver, and latent FM;
17. final recommended default CQ configuration.

The Stage-7 principle is:

> Spend extra capacity where it is paid once per condition, and add cheap explicit information/training stabilization, while keeping the repeated 128-D CQ query decoder and persistent Top-K inference path intact.

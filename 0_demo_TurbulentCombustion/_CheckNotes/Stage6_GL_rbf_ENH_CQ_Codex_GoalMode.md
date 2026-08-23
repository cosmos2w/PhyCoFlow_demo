# Stage 6 Goal-Mode Specification
## GL_rbf_ENH-CQ — Compact Query Decoder

### Repository / branch context

Repository: `cosmos2w/PhyCoFlow_demo`

Work from the latest local Stage-1–5 state on:

`perf/pointcloud-ffm-field-reconstruction`

Formal current-architecture reference:

- F0 best checkpoint: `F0_frozen_current_DemoN9300_20260821_075633/best.pt`
- F0 best epoch: 180
- F0 is the Stage-6 reference.
- Do not promote or replicate F1. Its 16,384-query protocol showed no material quality benefit while costing 3.56x training time and 1.85x sampled peak reserved GPU memory.

Stage 6 must isolate query-decoder architecture efficiency. Use F0-style supervision for model screening:

```yaml
n_query_points: 4096
train_query_microbatch_size: null
batch_size: 64
seed: 42
```

Do not mix the CQ architecture comparison with increased supervision.

---

# 1. Goal

Implement one new model family:

`GL_rbf_ENH_CQ` — **Compact Query Decoder**

The purpose is to reduce dominant per-query computation and activation memory while preserving the scientific identity of the current Global–Local RBF model.

The conceptual chain must remain:

```text
sparse observations
    ↓
sensor tokenization
    ↓
global latent reasoning + sensor re-injection
    ↓
globally enriched sensor tokens
    ↓
local top-K RBF gather
    ↓
global query context
    ↓
point-state / local / global fusion
    ↓
RF velocity
```

Stage-3 profiling already showed that at roughly 1M full points / 65,536 queries, data preparation is only about 45.7 ms while current model execution is about 709–801 ms. The bottleneck is now genuinely the repeated query decoder.

Reuse the Stage-1–5 execution infrastructure:

- selected training materialization;
- fixed-manifest evaluation;
- cached/streamed reconstruction;
- condition-context reuse;
- geometry/static query caches;
- query microbatching;
- legacy numerical reference paths.

---

# 2. Critical non-goals

Do not redesign the whole model.

For the primary CQ variants, do not change:

- latent dimension;
- number of latent slots;
- number of latent blocks;
- latent heads;
- latent re-injection schedule;
- sensor-token architecture;
- sensor Fourier coordinate encoding;
- field embeddings;
- sensor output projection;
- sensor back-attention / double-dip refinement;
- condition width;
- `gather_topk=32`;
- KeOps neighbor semantics;
- top-K RBF weighting;
- learnable RBF sigma;
- GLRES sensor-importance mechanism;
- RFF prior distribution;
- RF bridge;
- target `x1 - x0`;
- mean-MSE objective;
- Euler/Heun formula;
- observation/sensor-consistency rules;
- Stage-1–5 data-path behavior.

Do not reduce Fourier positional bands in the primary experiment.

Do not introduce anchors, voxel hierarchies, graph coarsening, neural operators, or a new latent field representation.

Do not use F1-style 16,384-query formal training during CQ screening.

---

# 3. Architectural principle

The current model spends too much compute at every query through:

1. a wide query-state MLP at width 256;
2. coordinate-based query-to-latent cross-attention;
3. a Transformer-style FFN in that query readout;
4. projection of query latent readout back to width 256;
5. a wide concatenation of approximately:
   - point feature: 256;
   - global/query-global feature: 256;
   - local RBF feature: 128;
6. a fusion head beginning from approximately 640 channels;
7. a GLRES coarse/scaffold branch that is also pointwise.

The CQ model keeps the same information sources but compresses their query-side execution.

Primary setting:

```yaml
cq_query_dim: 128
```

The sensor/global latent reasoning remains unchanged.

---

# 4. Add a sibling backbone; do not rewrite legacy GL_rbf_ENH

Add a new backbone choice:

```yaml
backbone: "GL_rbf_ENH_CQ"
```

Keep `GL_rbf_ENH` completely available and checkpoint-compatible.

Prefer a sibling class such as:

```python
ConditionalPointHybridLocalGlobalRBFCQ
```

Do not heavily refactor the current legacy class in the same task.

Reasons:

- F0 must remain an immutable reference;
- old checkpoints must keep loading strictly;
- CQ state dicts are intentionally different;
- Stage-6 changes must not risk legacy behavior;
- code deduplication can happen later if CQ is promoted.

Reusing existing helper modules/functions is fine when this does not alter legacy module names or semantics.

---

# 5. Preserve the F0 condition/global/local core

Inside `GL_rbf_ENH_CQ`, preserve the same configuration values and mathematics for:

## 5.1 Sensor branch

Keep field embedding, sensor Fourier encoding, sensor input projection, and mask handling.

## 5.2 Latent global processor

Keep learned latent array, input cross-attention, latent self-attention blocks, repeated sensor re-injection, summary extraction mode, latent dimension, and latent count.

## 5.3 Double-dip refinement

Keep sensor back-attention, masked refined sensor tokens, and `sensor_out_proj`.

## 5.4 Local query conditioning

Keep exact top-K search, K=32, KeOps, learnable RBF sigma, RBF weighting, GLRES sensor-importance bias, and all mask semantics.

No Stage-6 speed claim may come from changing K or neighborhood mathematics.

---

# 6. CQ query-state branch

Replace the 256-wide query point encoder with:

```python
cq_point_encoder = make_mlp(
    in_dim=coord_feat_dim + n_fields + 1,
    hidden_dim=cq_query_dim,
    out_dim=cq_query_dim,
    depth=3,
)
```

Primary:

```yaml
cq_query_dim: 128
```

Input remains exactly:

```text
[Fourier coordinates, x_t, t]
```

The intended change is learned query width, not the physical information given to each query.

---

# 7. CQ global-summary projection

The existing latent-summary path may remain at the F0 width internally.

Project it to the compact query space:

```python
cq_global_proj: hidden_dim -> cq_query_dim
```

Result:

```text
global_q: [B, cq_query_dim]
```

Do not create a full `[B, N, 256]` global tensor unnecessarily.

---

# 8. CQ local projection

The top-K RBF gather must produce the same local conditioning feature as F0.

If `cond_dim == cq_query_dim`, use an identity or minimal normalization path.

Otherwise use:

```python
cq_local_proj: cond_dim -> cq_query_dim
```

For the expected F0-compatible values (`cond_dim=128`, `cq_query_dim=128`), avoid unnecessary dense work.

---

# 9. CQ fusion: gated additive fusion

Do not construct the legacy approximately 640-wide concatenation.

Use:

```python
point_q = cq_point_encoder(...)
global_q = cq_global_proj(global_summary)
local_q = cq_local_proj(local_cond)
query_global_q = query readout result

fused = (
    point_q
    + cq_global_scale * global_q.unsqueeze(1)
    + cq_local_scale * local_q
    + cq_readout_scale * query_global_q
)

fused = cq_fusion_norm(fused)
```

Recommended scalar initializations:

```text
cq_global_scale  = 1.0
cq_local_scale   = 1.0
cq_readout_scale = 1e-2
```

Use learned scalars, not a heavy gating network.

---

# 10. Compact final velocity head

Recommended:

```python
cq_fusion_norm = nn.LayerNorm(cq_query_dim)

cq_head = nn.Sequential(
    nn.Linear(cq_query_dim, cq_query_dim),
    nn.GELU(),
    nn.Dropout(mlp_dropout),
    nn.Linear(cq_query_dim, cq_query_dim),
    nn.GELU(),
    nn.Dropout(mlp_dropout),
    nn.Linear(cq_query_dim, n_fields),
)
```

This preserves nonlinear residual capacity while avoiding the legacy `640 -> 256 -> 256 -> fields` path.

---

# 11. Preserve GLRES, but compact only its query-side coarse branch

Primary CQ must continue to support:

```yaml
gather_mode: "topk_rbf_glres"
```

Keep sensor importance exactly as in F0.

Replace only the query-side coarse scaffold with a compact equivalent:

```python
cq_coarse_film = nn.Linear(cq_query_dim, 2 * cq_query_dim)
```

Use compact global summary to produce gamma/beta:

```python
gamma, beta = cq_coarse_film(global_q).chunk(2, dim=-1)

coarse_feat = (
    point_q * (1 + torch.tanh(gamma).unsqueeze(1))
    + beta.unsqueeze(1)
)

coarse_pred = cq_coarse_scale * cq_coarse_head(coarse_feat)
```

Recommended `cq_coarse_head`:

```text
LayerNorm(128)
128 -> 128
GELU
128 -> n_fields
```

Initialize coarse scale using the same GLRES scale convention as F0.

---

# 12. Only two CQ variants

Add:

```yaml
cq_readout_mode: "full"
# choices: ["full", "lowrank"]
```

Experimental matrix:

```text
A — F0 reference
    GL_rbf_ENH
    256-wide query path
    concat fusion
    current full query-latent CrossAttentionBlock

B — CQ-Full
    GL_rbf_ENH_CQ
    query_dim=128
    additive fusion
    compact head/coarse branch
    full latent CrossAttentionBlock readout

C — CQ-LR
    GL_rbf_ENH_CQ
    query_dim=128
    additive fusion
    compact head/coarse branch
    low-rank lightweight latent readout
```

Do not add more primary variants.

---

# 13. CQ-Full latent readout

`cq_readout_mode="full"` is the conservative compact version.

Keep the current concept:

```text
coordinate query token
    ↓
CrossAttentionBlock against latent memory
    ↓
query-specific global feature
```

Use the same latent dimension and attention/FFN structure as current GL-RBF query readout.

Changes:

- query-side decoder token uses `cq_query_dim`;
- input projection ends at `latent_dim`;
- output projection is `latent_dim -> cq_query_dim`, not `latent_dim -> hidden_dim`.

This variant isolates savings from narrower query state, additive fusion, and compact heads.

---

# 14. CQ-LR latent readout

`cq_readout_mode="lowrank"` replaces only the expensive per-query Transformer-style readout.

Implement a dedicated module such as:

```python
CompactLatentReadout
```

Primary settings:

```yaml
cq_readout_rank: 64
cq_readout_heads: 4
```

Requirements:

- rank divisible by heads;
- `cq_query_dim` divisible by heads.

Recommended semantics:

```text
query Fourier feature -> Q projection, total rank 64 -> 4 heads
latents -> K projection, total rank 64 -> 4 heads
latents -> V projection, total value width 128 -> 4 heads
softmax(QK^T / sqrt(head_rank))
weighted latent values
concat heads -> [B, N, 128]
```

No Transformer query-side FFN.

No large per-query output projection is required initially.

A final LayerNorm on the 128-D readout is acceptable.

Critical optimization:

- projected latent K and V are condition-static;
- cache them once where Stage-4 context allows;
- do not recompute K/V per query chunk or ODE NFE.

The readout must remain query-specific latent access; do not replace it with only one broadcast global vector.

---

# 15. Stage-1–5 execution integration is mandatory

The CQ model must integrate with the existing local Stage-4/5 APIs.

Inspect actual current signatures and support:

- ordinary `forward(...)`;
- `prepare_condition_context(...)`;
- `prepare_query_context(...)`;
- `forward_query_chunk(...)`;
- cached/streamed reconstruction;
- cache levels `none`, `geometry`, `static_features`;
- differentiable condition reuse for query microbatch training;
- monolithic 4,096-query training;
- optional query microbatching;
- Euler and Heun;
- observation consistency modes.

Do not create separate CQ-only training/reconstruction infrastructure.

---

# 16. CQ cache semantics

Condition-static cache should reuse sensor tokens, latents, global summary, refined sensor features, and sensor-importance bias.

For CQ-LR also cache latent K/V projections.

Geometry cache remains exact current top-K indices and squared distances.

For inference, static-feature cache may contain:

- coordinate Fourier features;
- local RBF feature;
- compact query-global readout;
- other purely static compact inputs.

Never cache dynamic point-state features depending on `x_t` or `t`.

Report CQ static cache size at 1M queries.

---

# 17. Configuration

Add:

```yaml
# ----------------------------------------------------------
# Compact Query Decoder — GL_rbf_ENH_CQ
# ----------------------------------------------------------

backbone: "GL_rbf_ENH_CQ"

cq_query_dim: 128

cq_readout_mode: "lowrank"   # ["full", "lowrank"]
cq_readout_rank: 64
cq_readout_heads: 4

cq_global_scale_init: 1.0
cq_local_scale_init: 1.0
cq_readout_scale_init: 1.0e-2
```

Do not overload existing `hidden_dim`; legacy F0 checkpoint semantics must remain unchanged.

Clearly document condition/global-core dimensions versus CQ query-execution dimensions.

---

# 18. Checkpoint behavior

F0 loading must remain unchanged.

CQ checkpoints must store:

- `backbone=GL_rbf_ENH_CQ`;
- `cq_query_dim`;
- `cq_readout_mode`;
- `cq_readout_rank`;
- `cq_readout_heads`;
- gate initializations.

Do not pretend CQ is strictly checkpoint-compatible with F0.

Optional: add an explicit F0-to-CQ initializer that transfers only unchanged same-name/same-shape condition-core modules. Do not force mismatched query modules. This is diagnostic only.

Primary scientific CQ comparison should train from scratch with the same seed/protocol as F0 unless separately approved.

---

# 19. Stage-6 training protocol

Do not retrain F0 unless an artifact is missing.

Use the existing F0 best checkpoint for controlled evaluation and cost comparisons.

For CQ screens use:

```text
seed 42
batch size 64
n_query_points 4096
train_query_microbatch_size null
same dataset/split/optimizer/LR schedule/observations/RF objective as F0
```

Do not use F1's larger supervision.

---

# 20. Gate A — implementation and invariants

Before training:

1. legacy `GL_rbf_ENH` tests pass;
2. F0 checkpoint loads strictly;
3. CQ-Full forward works;
4. CQ-LR forward works;
5. output is `[B,N,n_fields]`;
6. `topk_rbf_glres` works;
7. learnable RBF sigma receives gradient;
8. sensor importance receives gradient;
9. latent/global modules receive gradient;
10. CQ point/readout/fusion/coarse modules receive gradient;
11. cached reconstruction works;
12. Stage-5 training APIs work.

Run the full regression suite.

---

# 21. Gate B — CQ self-equivalence

For both CQ-Full and CQ-LR:

## Reconstruction

Compare normal/full execution against cached-streamed execution using identical weights, RNG, and sparse condition.

Test:

- Euler NFE 1/2/4;
- Heun at least NFE 2;
- `topk_rbf_glres`;
- `default_hard`;
- `endpoint_smooth`.

Require tight FP32 agreement.

## Training

Compare monolithic versus query-microbatch execution on a small non-divisible query count.

Compare:

- loss;
- all gradients;
- RBF sigma gradient;
- one optimizer update.

Use Stage-5-like tolerances.

## Cache reuse

Assert:

- sensor/latent encoding once per reconstruction trajectory;
- CQ-LR K/V projections are not repeated per query chunk/NFE when cached;
- no legacy approximately 640-wide fused query tensor exists in CQ.

---

# 22. Gate C — cost benchmark before training

Benchmark F0, CQ-Full, CQ-LR on the same GPU-resident synthetic 3-D workload.

Required query sizes:

```text
4,096
16,384
65,536
```

Primary observations:

```text
M=256
```

Optionally verify `M=1024`.

Measure:

- forward;
- backward;
- training step;
- peak allocated/reserved memory;
- queries/sec;
- ms/1k queries;
- total parameters;
- query-decoder parameters.

Where practical, instrument:

- condition encoding;
- point encoder;
- latent readout;
- local gather;
- fusion/head;
- GLRES coarse branch.

Suggested screening targets:

CQ-Full:
- >=1.4x model-step speedup at 65,536 queries;
- >=25% lower query-model peak allocation.

CQ-LR:
- >=1.7x speedup;
- >=35% lower peak allocation.

If CQ-LR is not meaningfully faster than CQ-Full, inspect implementation before training.

---

# 23. One-million-point reconstruction benchmark

Compare F0, CQ-Full, CQ-LR with cached-streamed Euler NFE=2:

```text
N_query=1,000,000
batch=1
same M/K
same query chunk where possible
```

Report:

- wall time;
- sec / million queries / NFE;
- peak allocated/reserved memory;
- static cache size;
- dynamic peak.

Reference Stage-4 F0 result:

```text
2.675 s at NFE=2
~2,958 MB peak allocation
~2,197 MB FP32 static cache
```

Use same-run relative comparisons; do not require exact reproduction of historical timings.

---

# 24. Gate D — limited training screen

After correctness and cost gates pass, train only CQ-Full and CQ-LR.

Do not retrain F0.

Use:

```text
60 epochs
seed 42
batch size 64
n_query_points 4096
no query microbatch
same optimizer/scheduler as F0
```

Use fixed-manifest evaluation around epochs 1, 20, 40, 60.

At epoch 60 save reconstruction on the same controlled validation snapshot/sensor layout used for F0 formal baseline:

```text
Euler NFE 1/2/4
```

This is candidate selection, not publication accuracy.

---

# 25. Candidate selection

Promote CQ-LR to the 200-epoch formal run if:

- epoch-60 fixed-manifest RF loss is within about 3% of CQ-Full;
- reconstruction mean is within about 3% of CQ-Full;
- and CQ-LR gives at least about 15% additional model speedup over CQ-Full.

Otherwise promote CQ-Full.

Do not launch a third architecture unless both compact variants clearly fail.

---

# 26. Single rescue configuration only if both fail badly

If both CQ-Full and CQ-LR appear under-capacity, allow exactly one rescue configuration using the same CQ code:

```yaml
cq_query_dim: 160
cq_readout_mode: "full"
```

Only use this if both compact variants are roughly >5% worse in controlled validation and reconstruction.

Run only a short screen.

Do not simultaneously alter latent count, K, Fourier bands, or global architecture.

If CQ-160 also fails materially, stop and report that the original 256-wide query capacity appears important.

---

# 27. Formal Stage-6 run after selection

Prepare the formal config/launcher, but do not launch 200 epochs unless long-run execution is explicitly allowed.

Train selected CQ from scratch using:

```text
seed 42
200 epochs
batch size 64
n_query_points 4096
same F0 dataset/observations/optimizer/LR schedule
```

Compare selected CQ best checkpoint to F0 best with the established protocol:

## Fixed-manifest RF evaluation

- same 64 layouts;
- same three RF repeats/layout;
- paired layout-level CI.

## Matched reconstruction

Same:

- validation snapshot;
- 256 T sensors;
- sensor checksum;
- RF sample seed;
- Euler;
- cached streaming;
- NFE 1/2/4.

Report per-field and five-field mean relative L2 plus hard-sensor error.

## Efficiency

Report:

- train time/epoch;
- step time;
- peak memory;
- model-only 65,536-query benchmark;
- 1M-query NFE-2 reconstruction.

---

# 28. Formal promotion rule versus F0

Strong promotion:

```text
>=1.7x model execution speedup at 65,536 queries
AND
fixed-manifest RF loss <= +2% relative to F0
AND
five-field reconstruction mean <= +2% relative to F0 at NFE 1/2/4
```

Conditional promotion:

If execution speedup is >=2x, a small explicit reconstruction tradeoff up to about 2–3% may be acceptable.

Reject if:

- speed improvement is small;
- reconstruction is consistently >3–5% worse;
- or an important field degrades strongly even if the mean looks acceptable.

Pay special attention to `U_1`.

---

# 29. Diagnostics and reporting

Add a model summary containing:

- total parameter count;
- condition-core parameter count;
- query-decoder parameter count;
- query_dim;
- latent_dim;
- cond_dim;
- readout mode;
- rank/heads;
- major query tensor widths;
- legacy concat width;
- CQ fused width.

If practical, include simple MAC estimates for query-side linear layers, clearly labeled as theoretical estimates rather than runtime.

---

# 30. Naming / evidence directory

Use:

```text
GL_rbf_ENH_CQ
CQ-Full
CQ-LR
```

Evidence:

`0_demo_TurbulentCombustion/_CheckNotes/Stage6_compact_query/`

Suggested subdirectories:

```text
implementation/
benchmarks/
equivalence/
screen_cq_full/
screen_cq_lr/
formal_candidate/
```

Central report:

`_CheckNotes/Stage6_compact_query/README.md`

---

# 31. Tests

Add `tests/test_pointcloud_cq.py`.

Cover at minimum:

1. CQ config validation;
2. CQ-Full output shapes;
3. CQ-LR output shapes;
4. rank/head validation;
5. `topk_rbf_glres`;
6. learnable sigma gradient;
7. sensor/global latent gradients;
8. CQ decoder gradients;
9. full vs cached-streamed equivalence;
10. Euler;
11. Heun;
12. endpoint-smooth;
13. monolithic vs microbatch gradient equivalence;
14. CQ-LR K/V cache reuse;
15. large-query chunk smoke;
16. legacy F0 strict-load regression.

Run the complete regression suite afterward.

---

# 32. Do not clean up legacy code in this task

Retain:

- F0 `GL_rbf_ENH`;
- legacy full reconstruction;
- cached-streamed reconstruction;
- monolithic training;
- microbatch training;
- data-path legacy options required by tests.

Cleanup comes only after a CQ candidate is formally accepted.

---

# 33. Goal-mode execution order

Work continuously:

```text
Gate A
    implement CQ sibling model + config + tests

Gate B
    integrate Stage-4/5 execution
    prove CQ self-equivalence

Gate C
    run scaling and 1M reconstruction benchmarks

Gate D
    run 60-epoch CQ-Full and CQ-LR screens
    select one candidate

Final
    prepare formal 200-epoch candidate config/launcher
    do not launch it unless long-run execution is explicitly permitted
```

Do not create additional architecture stages.

---

# 34. Final deliverables

Provide:

1. commit SHA(s);
2. files changed;
3. exact CQ config surface;
4. architecture summary;
5. confirmation F0 legacy model is untouched;
6. confirmation sensor/latent/local-RBF mathematics remain unchanged;
7. full regression-suite result;
8. CQ-Full vs CQ-LR parameter counts;
9. 4k/16k/65k model benchmark;
10. 1M-query NFE-2 reconstruction benchmark;
11. cached-streamed equivalence;
12. query-microbatch equivalence;
13. 60-epoch screen results;
14. matched epoch-60 reconstruction;
15. selected candidate and rationale;
16. prepared 200-epoch formal config;
17. remaining bottleneck analysis.

Also identify where the measured speedup came from:

```text
point-state width
fusion/head width
latent readout
GLRES coarse branch
other
```

The objective is not simply to reduce parameter count. It is to determine whether the **Global–Local RBF design can preserve reconstruction quality while making the repeated query decoder substantially cheaper**, enabling the same method to scale naturally from the current 2-D demonstration to million-point 3-D fields.

# Next Goal-Mode Round — CQ-Balanced Quality Recovery + Safe Efficiency

Repository: `cosmos2w/PhyCoFlow_demo`

Verified remote tip: `d95b7803ad2e0850072369fc9d9928a024c7e7e9`

Both remote branches currently resolve to this SHA:
- `perf/pointcloud-ffm-query-decoder-stage6`
- `perf/pointcloud-ffm-field-reconstruction`

Recommended new branch:
`perf/pointcloud-cq-balanced`

## Current status

F0 remains the scientific baseline.

| Metric | F0-ENH | CQ-LR | CQ-LR change |
|---|---:|---:|---:|
| Mean epoch time | 27.150 s | 18.294 s | -32.62% |
| Diagnostic train step | 581.410 ms | 379.276 ms | -34.77% |
| Peak allocated | 27,642.9 MiB | 23,258.3 MiB | -15.86% |
| Peak reserved | 36,414.0 MiB | 27,688.0 MiB | -23.96% |
| Best validation loss | 0.353095 | 0.388921 | +10.15% |
| Final validation loss | 0.361207 | 0.400808 | +10.96% |

CQ-LR is a real efficiency success but is too aggressive scientifically to replace F0.

Persistent geometry-only Top-K is validated and should remain enabled for repeated fixed-geometry inference:

- 1M queries, Euler NFE=4:
  - Stage-4 per-call static cache: 0.4775 s
  - persistent geometry + static cache: 0.3579 s
  - 1.33x faster
  - geometry build: 0.1458 s once
  - geometry storage: 396.7 MiB
  - post-build KNN calls: 0
  - max output difference: 1.907e-6

## Diagnosis

The CQ implementation correctly preserves the F0 condition/global/local core. CQ-LR changes several query-side properties simultaneously:

1. query width 256 -> 128;
2. global feature projected to 128;
3. local feature enters the same 128-D fused space;
4. full latent readout replaced by rank-64 low-rank readout;
5. F0 structured concatenation replaced by early additive fusion:
   `point + global + local + query-global`;
6. final head works only on the 128-D fused state;
7. GLRES coarse branch is also reduced.

Stage-6 evidence suggests low-rank readout is only part of the quality loss: CQ-Full-128 was modestly better than CQ-LR-128, and CQ-160/full improved further, but neither recovered F0.

Primary hypothesis:

> The main scientific over-compression is likely the combination of narrow query capacity and early additive fusion, not the preserved GL-RBF condition/global/local core.

## Goal

Build one conservative recovery variant:

# CQ-Balanced-192-Full

Keep backbone `GL_rbf_ENH_CQ` and add a backward-compatible fusion option.

Primary config:

```yaml
backbone: "GL_rbf_ENH_CQ"
cq_query_dim: 192
cq_readout_mode: "full"
cq_fusion_mode: "structured_concat"

gather_mode: "topk_rbf_glres"
gather_topk: 32
latent_dim: 128
num_latents: 128
num_heads: 8
num_latent_blocks: 4
cond_dim: 128
```

Do not change sensor/global/local core mathematics.

## Hard invariants

Do not change:
- sensor encoder
- field embedding
- Fourier coordinate encoding
- latent width/count/blocks/heads
- latent reinjection
- sensor back-attention/refinement
- condition width 128
- K=32
- KeOps
- RBF weighting
- learnable sigma
- sensor importance
- RFF prior
- RF objective
- query count 4096
- optimizer/schedule
- Stage-1–5 data path
- persistent Top-K
- solver/observation consistency

Do not use the old F1 16,384-query protocol.

## Backward compatibility

Add:

```yaml
cq_fusion_mode: "additive"
# choices: ["additive", "structured_concat"]
```

Default must remain `additive` if absent so existing CQ-LR/CQ-Full checkpoints remain reconstructible.

Do not alter existing CQ module names/shapes for `additive`.

Add fusion mode to checkpoint metadata and compatibility diagnostics.

## Structured fusion design

### Query state
Use width 192 with the same input:
`[Fourier coordinates, x_t, t]`.

### Global query context
Use full latent readout and restore the F0 semantic pattern:

```python
global_q = cq_global_proj(global_feat)             # [B,192]
query_global_q = full latent readout               # [B,N,192]
global_for_head = global_q.unsqueeze(1) + cq_readout_scale * query_global_q
```

### Local condition
Keep local RBF output at its native 128-D condition width. Do not project it to 192 merely for addition.

### Structured concatenation
Use:

```python
head_in = torch.cat([point_q, global_for_head, local_cond], dim=-1)
```

Fusion width:
- F0: 640
- CQ-LR additive: 128
- CQ-Balanced: 512

### Head
For structured concat:

```text
LayerNorm(512)
512 -> 192
GELU
192 -> 192
GELU
192 -> n_fields
```

### GLRES coarse branch
Keep GLRES at query width 192:
- film 192 -> 384
- coarse head 192 -> 192 -> fields
- sensor importance unchanged

## Why 192

Do not sweep many widths.

Existing evidence:
- 128 too compressed
- 160/full/additive improved but remained behind F0

Choose 192 as a conservative midpoint while also restoring information separation.


## Persistent Top-K

The new candidate must preserve:
- `prepare_reconstruction_geometry_cache()`
- persistent geometry validation
- zero new KNN after cache construction
- geometry/static_features modes
- CQ cached-streamed execution

Do not change persistent-cache semantics during this architecture test.

## Gate A — implementation

Before training:
1. F0 strict-load tests pass
2. old CQ-LR configs/checkpoints remain valid
3. additive CQ path unchanged
4. structured-concat forward works
5. output shapes correct
6. learnable sigma gradient correct
7. all preserved core gradients reachable
8. CQ new head/readout/coarse gradients reachable
9. persistent Top-K tests pass
10. cached-streamed equivalence passes
11. query-microbatch equivalence passes
12. full regression suite passes

Current PR reports 116 passing tests; do not reduce coverage.

## Gate B — self-equivalence

For CQ-Balanced compare normal/full and cached-streamed:

- Euler NFE 1/2/4
- Heun NFE 2
- cache geometry/static_features
- persistent geometry off/on

Use tight FP32 tolerances.

Also verify monolithic vs query-microbatch loss, all gradients, sigma gradient, and one optimizer update.

## Gate C — cost benchmark

Compare:
- F0-ENH
- CQ-LR-128
- CQ-Balanced-192-Full

Query sizes:
- 4,096
- 16,384
- 65,536

Primary M=256.

Record:
- forward
- backward
- optimizer/full step
- peak allocated/reserved
- total/query params
- component timings

Also benchmark repeated inference:
- 1M queries
- Euler NFE=4
- persistent geometry + static_features

If feasible include 125^3 queries.

### Efficiency gate

Require at least:
- >=1.15x full train-step speedup vs F0
- >=10% lower peak allocated or reserved memory vs F0

Desirable:
- >=1.20x step speedup
- >=15% memory reduction

Persistent inference must still work and should not regress >10% vs F0 persistent inference.

## Gate D — 200-epoch resumable screen

Do not immediately spend 1000 epochs.

Train CQ-Balanced-192-Full for 200 epochs using the clean A/B protocol:

```text
seed 42
batch 128
n_query_points 4096
monolithic queries
lr 1e-4
same weight decay
scheduler_t_max 1000
same data/split/stats
same observation distribution
same RF objective
```

Run 200 epochs initially but keep scheduler horizon 1000.

Save:
`1,20,40,60,100,150,200`.

Use the existing fixed 64-layout manifest and RF repeats.

Do not retrain F0 or CQ-LR; compare to saved milestone checkpoints.

## 200-epoch continuation gate

Continue the same run to 1000 only if:

### Quality
- fixed-manifest RF gap vs F0 at epoch 200 <=5%, OR at least 50% of the F0-vs-CQ-LR gap is recovered
- validation curve is materially closer to F0 than CQ-LR
- matched reconstruction does not broadly degrade >5%
- no major field, especially U_1, has a severe new failure

### Efficiency
- mean epoch time >=15% faster than F0
- step >=15% faster
- peak memory >=10% lower

If quality fails but efficiency passes, run only the CQ-224 fallback.

## If 192 passes

Resume the same run from epoch 200 to 1000; do not restart.

Evaluate fixed manifest at:
`400,600,800,1000`.

## Final promotion criteria

Strong promotion target relative to clean F0:
- best validation <= +2.5%
- fixed-manifest RF <= +2.5%
- mean reconstruction NFE 1/2/4 <= +3%
- no major single field >5% worse
- epoch time >=15% faster
- peak GPU memory >=10% lower

If quality is within ~1% with >=15% efficiency gain, promote CQ-Balanced as the default ENH-CQ scientific model.

If quality is 2.5–5% worse but efficiency >=20% better, keep it as a performance/3-D candidate and retain F0 as highest-quality reference.

If >5% quality gap remains, do not promote.

## Do not revisit larger supervision

Keep `n_query_points=4096`.

F1 already showed no benefit from 16,384 effective queries at major cost.

## Kernel-only training acceleration after architecture selection

Training profiling indicates backward + optimizer dominates. Treat kernel work separately from scientific architecture.

### Masked-attention benchmark

Using the same attention weights compare:
- existing `nn.MultiheadAttention` + boolean `key_padding_mask`
- explicit SDPA path using the same q/k/v/out parameters
- alternative additive mask if relevant

Use actual F0/CQ-Balanced shapes and H100/BF16 when available.

Measure:
- attention forward
- attention backward
- full model forward
- full training step

Verify:
- outputs
- input gradients
- all attention parameter gradients
- one optimizer step

Only add an `attention_backend` switch if:
- attention is materially faster
- full training step >=8% faster
- numerical parity passes

Default legacy backend remains available.

Do not mix a new backend into the first CQ-Balanced scientific run unless parity is clearly established.

### Optional fused AdamW

Only if optimizer-only profiling is material.

Compare current / fused / foreach AdamW with identical gradients and hyperparameters. Adopt only if end-to-end benefit is meaningful and one-step numerical parity is acceptable.

Do not delay CQ-Balanced for this.

## Inference recommendation during this round

For repeated fixed-geometry inference / CRPS:

```text
build persistent geometry once
reconstruction_cache_level = static_features
Euler NFE = 4
```

For the final candidate report:
- one-time geometry build
- steady NFE2 latency
- steady NFE4 latency
- marginal NFE4 cost over NFE2
- CRPS if available

Do not recompute Top-K per ensemble member.

## Evidence package

Create:

`_CheckNotes/Stage6_CQ_balanced_quality_recovery/`

Suggested:
- README.md
- implementation/
- cost_benchmark/
- screen_200/
- formal_1000/
- kernel_benchmark/
- figures/

Generate:
1. validation loss vs epoch for F0/CQ-LR/CQ-Balanced
2. fixed-manifest RF vs epoch
3. epoch time + peak memory
4. matched reconstruction NFE1/2/4
5. quality-efficiency Pareto
6. kernel timing before/after if used

## Branch / PR hygiene

Create a child branch from `d95b780`.

Prefer a separate draft PR for CQ-Balanced rather than expanding PR #1 further.

Do not delete old configs/results.

## Final deliverables

Report:
1. branch and SHA
2. files changed
3. `cq_fusion_mode` implementation
4. old CQ checkpoint compatibility
5. regression count
6. parameter counts
7. 4k/16k/65k cost results
8. persistent Top-K inference result
9. 200-epoch screen
10. continuation decision
11. if continued, 1000-epoch result
12. F0/CQ-LR/CQ-Balanced Pareto
13. masked-attention result
14. fused-optimizer result if performed
15. final recommendation for:
   - quality model
   - throughput model
   - formal 3-D model

Central question:

> Can restoring structured point/global/local information separation at a moderate 192-D query width recover F0 quality while retaining a meaningful fraction of CQ's efficiency gain?

Do not open another broad architecture search until this single hypothesis has been tested cleanly.
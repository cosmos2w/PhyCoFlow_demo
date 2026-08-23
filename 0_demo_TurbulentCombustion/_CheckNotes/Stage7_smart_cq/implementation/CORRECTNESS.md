# Stage-7 implementation and correctness evidence

## Backward compatibility

All new architecture features default to the frozen CQ-v1 behavior:

```yaml
model_ema_enabled: false
model_ema_decay: 0.999
model_ema_eval: true
cq_time_conditioning: scalar_concat
cq_time_embed_dim: 128
cq_time_max_period: 10000.0
cq_time_film_zero_init: true
cq_measurement_support_mode: none
cq_measurement_support_normalize: true
```

When Stage-7 architecture features are disabled, no FiLM or measurement/support modules are instantiated and the historical CQ head remains `128 -> 128 -> 128 -> fields`. The clean CQ-LR-128 checkpoint at `_CheckNotes/Stage6_clean_ab/runs/CQ_LR_1K_B128_DemoN9511_20260821_235104/best.pt` strict-loaded with no missing or unexpected keys.

## EMA

`ModelEMA` tracks parameters and buffers after every optimizer step. Checkpoints contain live `model` state and independent `model_ema` state, decay, and update count. Resume explicitly restores live weights for training and then restores the EMA shadow. Offline evaluation selects EMA weights only when checkpoint metadata enables EMA evaluation. Validation/reconstruction use a context manager and restore live weights exactly afterward.

## Time FiLM

The scalar `t` remains in the historical point-encoder input. Optional sinusoidal/MLP time conditioning applies standard residual FiLM, `point_q * (1 + scale) + shift`. The final projection is zero-initialized by default, making the enabled branch an exact identity at initialization. No time enters sensor tokens, latent encoding, refined sensors, persistent geometry, or coordinate/readout caches.

## Raw measurement/support

The shortcut retains raw normalized `obs_values` and `obs_field_ids` only in the enabled condition context. A single Top-K result feeds both learned local conditioning and explicit statistics. Base weights are a softmax of geometry-only RBF logits; GLRES sensor-importance bias is intentionally excluded. Differentiable field-wise scatter accumulation avoids a dense `[B,Q,K,field]` tensor. For each field, support is its share of total Top-K RBF weight and value is the within-field weighted mean. Missing fields return zero value/support.

The uncached combined path asks the existing KNN routine for geometry only, computes the explicit statistics, then gathers learned sensor features once. This changes allocation order without adding a search. `static_features` caches learned local conditioning, compact latent readout, and the 10-scalar value/support feature. Persistent geometry execution performs zero KNN calls after geometry construction.

## Exact-gradient execution microbatch

The formal Stage-7 configs use `train_query_microbatch_size: 2048` with one reused condition context. The RF source, timestep, target, B128 batch, and 4096-query objective are unchanged. The regression suite checks monolithic versus microbatch loss, gradients, and optimizer updates for both historical and all-on Stage-7 paths.

## Commands and results

```text
pytest -q tests/test_stage7_smart_cq.py
11 passed

pytest -q tests/test_pointcloud_cq.py tests/test_cq_balanced.py \
  tests/test_cq_persistent_topk_cache.py tests/test_pointcloud_query_microbatch.py
84 passed, 1 skipped

pytest -q
141 passed, 1 skipped
```

Focused coverage includes EMA formula/apply/restore/checkpoint/resume, zero-init FiLM identity and gradients, literal latent widths 128/256 with a fixed 128-D query decoder, hand-computed value/support, sigma gradient, changed values/field IDs, one-search uncached execution, one condition-context build over cached NFE-4, zero post-build persistent KNN, cached/fresh equivalence, and monolithic/query-microbatch loss-gradient-update equivalence for Stage-7 off/all-on.

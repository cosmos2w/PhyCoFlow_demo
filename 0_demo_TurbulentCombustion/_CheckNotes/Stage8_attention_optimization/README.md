# Stage 8 — GL_rbf_CQ attention execution optimization

Branch: `perf/stage8-attention-optimization`

Frozen numerical/checkpoint oracle: `gl-rbf-cq-v0.9.0-rc1` and the portable
release checkpoint with SHA256
`2516ffeb45775d4e6b8d88b4b24d927aac28665a2a90102583e07deaca78f64d`.

## Implementation

The public GL_rbf_CQ default is cached K/V with full padding. The historical MHA
path remains supported, and unpromoted research modes remain available without
adding parameters or state-dict keys:

```yaml
condition_attention_execution: cached_kv
sensor_attention_padding_mode: full

# Compatibility/debug override:
# condition_attention_execution: legacy_mha
```

`cached_kv` normalizes and projects sensor K/V plus the padding mask once per
condition (or once per active static bucket), while every latent re-injection
still computes its own normalized Q, attention output, residual, and FFN. The
shared graph remains differentiable, so gradients from all four reads accumulate
through sensor tokens and K/V weights.

Static bucketing first verifies that valid sensors are a contiguous prefix. It
groups samples by the smallest configured bucket, scatters sensor-to-latent
outputs back to batch order, and runs every latent self-attention block once on
the complete batch. Sensor-back-attention evaluates only the bucket-length query
prefix and pads it back to the original sensor-slot layout. A non-prefix mask
falls back to full padding.

No RF, model parameter, Top-K/RBF/GLRES, EMA, microbatch, persistent-cache, or
solver code was changed.

## Correctness

- All modes strict-load the release checkpoint: 148 keys and schema SHA256
  `f221ee2b26268fe64e116f9f57165d34aac9ff524202e40d225ed17f76dc970e`.
- Focused Stage-8 suite: 7 passed.
- Final RC2 complete regression suite: 162 passed.
- Frozen-checkpoint FP32 maximum differences versus legacy/full:
  - cached/full output `3.34e-6`, context `4.29e-6`, gradients `5.07e-7`;
  - cached/bucketed output `5.25e-6`, context `4.29e-6`, gradients `4.77e-7`.
- AdamW update-vector relative L2 difference was 0.133% for cached/full and
  0.129% for cached/bucketed. The maximum element is an attention projection
  bias whose theoretical gradient is zero; AdamW amplifies FP32 cancellation
  noise there. EMA-shadow relative L2 differences remained about `1.2e-9`.
- Unit gates cover all input/attention parameter gradients, four repeated
  re-injections, mixed bucket boundaries, the full RF loss and update,
  query-microbatch equivalence, EMA, Euler/Heun, and persistent Top-K with zero
  post-build KNN calls.

Machine-readable evidence: `correctness.json`.

Final default-selection and RC1 checkpoint evidence is in
`rc2_compatibility.json`. Public-default and historical-fallback focused tests
pass together with the Stage-8 suite (`19 passed`); the complete RC2 suite is
`162 passed`.

## B128/Q4096 benchmark

RTX 6000 Ada GPU 1, M=384 with valid counts 192–384, L=128, D=256, 8 heads,
4 latent blocks, query microbatch 2048, three warmups and eight measured AdamW
steps. Times are medians.

| mode | K/V projections | condition context (ms) | whole step (ms) | step change | peak allocated (MiB) | memory change |
|---|---:|---:|---:|---:|---:|---:|
| A legacy/full | 4 | 28.298 | 363.164 | reference | 20761.6 | reference |
| B cached/full | 1 | 24.245 | 339.993 | 6.38% faster | 20261.4 | 2.41% lower |
| C cached/[256,320,384] | 3 | 25.398 | 344.349 | 5.18% faster | 20330.0 | 2.08% lower |
| cached/[288,384] | 2 | 25.589 | 349.428 | 3.78% faster | 20352.1 | 1.97% lower |
| cached/[224,288,336,384] | 4 | 26.184 | 352.945 | 2.81% faster | 20319.6 | 2.13% lower |
| dynamic per-count diagnostic | 128 | 251.026 | 1399.074 | 285.25% slower | 20219.6 | 2.61% lower |

Cached/full reduced condition-context time by 14.32%. Static buckets reduced
padded arithmetic but lost more to sub-batch indexing/scattering and extra
kernels; no tested bucket policy beat cached/full. The dynamic diagnostic
confirms that fully dynamic per-count execution is unsuitable at B128.

Evidence: `benchmark.csv/json`, `bucket_comparison.csv/json`, and the best-mode
Chrome trace in `profiler/B_cached_full.json`.

## Smoke and pre-long-run decision

The three-epoch paired real-data smoke was stable and produced virtually
identical EMA validation loss. Cached/full was 3.48% faster per training epoch;
an epoch-4 resume successfully restored model, EMA, optimizer, scheduler, and
history. See `smoke/RESULTS.md`.

The Stage-8 promotion threshold requires at least 8% whole-step speedup. The
best mode, cached/full, reaches 6.38% in the controlled benchmark and 3.48% in
the real-data smoke, so it does **not** qualify for a later long validation run.

At this intermediate gate, cached/full was the only mode retained for a longer
quality validation. Static buckets and dynamic trimming were rejected.

## Long cached/full validation through epoch 600

The optional cached/full run was subsequently continued and intentionally
stopped after its complete epoch-650 validation. Its formal matched endpoint is
epoch 600. Against Stage7-All256 at epochs 200/400/600:

- fixed-manifest RF is 3.07% worse at epoch 200, 0.53% worse at epoch 400, and
  0.07% better at epoch 600; the paired epoch-600 95% interval includes zero;
- matched epoch-600 reconstruction is 2.21% better at Euler NFE1 and 1.67%
  better at Euler NFE4 on the shared deterministic snapshot;
- mean real training time over epochs 250–600 is 6.11% lower;
- the controlled benchmark remains 6.38% faster with 2.41% lower peak
  allocation;
- EMA decay, update counts, evaluation selection, and EMA/live contraction
  match the Stage7 behavior.

This passes the mature-quality, reconstruction, EMA, and directional-efficiency
checks, but still does not meet the original 8% promotion-speed target. See
`long_cached/evaluation_0600/RESULTS.md` and its machine-readable companions.

## Final selection

Promote `cached_kv + full` as the recommended/default GL_rbf_CQ execution. The
long validation confirms that the smaller-than-planned efficiency gain is real
and does not damage mature RF or reconstruction quality. Keep `legacy_mha +
full` fully supported for historical reproduction and numerical debugging. Do
not promote static bucketing or dynamic trimming.

This selection changes no scientific architecture, parameter, state-dict key,
RF objective, cache semantics, solver behavior, or release weights. The Stage-7
epoch-1000 portable checkpoint remains the release model artifact.

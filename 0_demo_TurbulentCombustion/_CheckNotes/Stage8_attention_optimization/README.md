# Stage 8 — GL_rbf_CQ attention execution optimization

Branch: `perf/stage8-attention-optimization`

Frozen numerical/checkpoint oracle: `gl-rbf-cq-v0.9.0-rc1` and the portable
release checkpoint with SHA256
`2516ffeb45775d4e6b8d88b4b24d927aac28665a2a90102583e07deaca78f64d`.

## Implementation

Three YAML-controlled execution modes are available without adding parameters
or state-dict keys:

```yaml
condition_attention_execution: legacy_mha  # legacy_mha | cached_kv
sensor_attention_padding_mode: full        # full | static_buckets
sensor_attention_buckets: [256, 320, 384]
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
- Complete regression suite: 161 passed.
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

## Smoke and recommendation

The three-epoch paired real-data smoke was stable and produced virtually
identical EMA validation loss. Cached/full was 3.48% faster per training epoch;
an epoch-4 resume successfully restored model, EMA, optimizer, scheduler, and
history. See `smoke/RESULTS.md`.

The Stage-8 promotion threshold requires at least 8% whole-step speedup. The
best mode, cached/full, reaches 6.38% in the controlled benchmark and 3.48% in
the real-data smoke, so it does **not** qualify for a later long validation run.

Recommendation for the later 1–2 long runs: retain `legacy_mha + full` as the
validated production execution. Keep `cached_kv + full` as the only promising
optional Stage-8 mode for short profiling or if the promotion threshold is
explicitly relaxed. Do not promote static buckets or dynamic trimming on this
hardware/workload.

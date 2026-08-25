# Stage 8 — Attention Execution Optimization for GL_rbf_CQ

## Goal

Optimize the validated `GL_rbf_CQ` attention execution without changing its scientific model.

Stage 8 targets three execution inefficiencies in the condition encoder:

1. repeated sensor K/V projection across latent re-injection;
2. repeated handling of the same sensor padding mask;
3. unnecessary attention/FFN work on padded sensor slots.

This is an execution optimization round only. No long training is required.

---

## 1. Scientific invariants

Do not change:

- RF bridge, prior, loss, or target;
- latent width/count/depth;
- CQ query width/readout/fusion;
- sensor embeddings;
- K=32, KeOps, RBF sigma, GLRES;
- Stage-7 FiLM and measurement/support shortcut;
- EMA;
- query microbatch mathematics;
- persistent Top-K;
- Euler/Heun or observation consistency.

The frozen RC implementation remains the numerical oracle.

---

## 2. Optimization A — cache sensor K/V across latent re-injection

Current sensor→latent attention repeatedly receives the same sensor tokens and mask while only the latent queries change.

Instead of recomputing

\[
\bar H_s=\mathrm{LN}_{KV}(H_s),
\]

\[
K_s=\bar H_sW_K,\qquad
V_s=\bar H_sW_V
\]

at every re-injection, prepare \(K_s\), \(V_s\), and the padding mask once per condition and reuse them.

For each latent re-injection \(r\), only compute

\[
Q_r=\mathrm{LN}_{Q}(Z_r)W_Q
\]

followed by the same attention:

\[
A_r=
\operatorname{softmax}
\left(
\frac{Q_rK_s^\top}{\sqrt{d_h}}+M_s
\right),
\]

\[
O_r=A_rV_s.
\]

Requirements:

- reuse existing attention weights/state-dict keys;
- no new trainable parameters;
- no detaching/no-grad;
- gradients from all re-injections must accumulate through the shared K/V graph;
- preserve the legacy MHA path as an option.

Suggested config:

```yaml
condition_attention_execution: legacy_mha
# legacy_mha | cached_kv
```

---

## 3. Optimization B — static sensor-length buckets

The optimized loader already pads to a fixed sensor maximum, but valid sensor count varies.

Avoid fully dynamic attention lengths. Instead use a small number of static buckets.

Example:

```yaml
sensor_attention_padding_mode: full
# full | static_buckets

sensor_attention_buckets: [256, 320, 384]
```

For each sample:

```text
bucket = smallest configured size >= valid sensor count
```

Group samples by bucket.

At each sensor→latent re-injection:

1. select the batch samples in each bucket;
2. run cross-attention using only that bucket length;
3. scatter outputs back to original batch order;
4. run latent self-attention once on the full batch.

Do not split latent self-attention into bucketed kernels.

Assume valid sensors form a contiguous prefix only after verifying it. If not, safely fall back to full padding.

---

## 4. Optimization C — avoid padded sensor-back-attention

The current sensor-back-attention processes padded sensor queries and masks them afterward.

Reuse the same sensor bucket partition:

```text
valid/padded sensor queries
        ↓
run back-attention only to bucket length
        ↓
pad result back to original M_max layout
```

Preserve the original sensor-slot ordering because downstream Top-K indices and field IDs depend on it.

---

## 5. Do not modify these attention paths

Leave unchanged:

- latent self-attention blocks;
- CQ low-rank query-to-latent readout;
- persistent Top-K query execution.

They already use fixed or cache-friendly shapes and are not the Stage-8 target.

---

## 6. Required execution modes

Support three direct comparison modes:

```text
A. legacy_mha + full padding
B. cached_kv + full padding
C. cached_kv + static_buckets
```

Optionally benchmark a dynamically trimmed attention path only to investigate the coworker's reported speedup.

Do not make fully dynamic sequence length the production default.

---

## 7. Correctness tests

Before performance claims, verify:

### Attention-block parity

Compare legacy MHA vs cached-K/V execution for:

- forward output;
- input gradients;
- all attention parameter gradients.

### Repeated re-injection parity

Use one prepared sensor K/V in four re-injections and compare against four legacy calls.

Verify gradients to:

- sensor tokens;
- Q/K/V projections;
- output projection;
- FFN.

### Bucket parity

Use mixed sensor counts around bucket boundaries.

Compare full padded vs bucketed execution for:

- condition latents;
- global summary;
- refined sensor features;
- full RF output;
- gradients.

### Whole-model parity

Using the frozen `GL_rbf_CQ` checkpoint, compare all three execution modes with identical:

- data;
- observations;
- RF source;
- \(t\);
- query layout;
- RNG state.

Check:

- loss;
- all parameter gradients;
- one AdamW update;
- EMA update;
- query-microbatch equivalence;
- Euler/Heun reconstruction;
- persistent Top-K with zero post-build KNN.

Old checkpoints must strict-load.

---

## 8. Real-shape benchmark

Use the validated training shape:

```text
B=128
Q=4096
query microbatch=2048
M_max=384
valid sensors=192–384
L=128
D=256
heads=8
latent blocks=4
```

Benchmark:

```text
A. legacy_mha + full
B. cached_kv + full
C. cached_kv + static_buckets
D. dynamic trim (diagnostic only)
```

Measure separately:

- sensor tokenization;
- sensor→latent attention;
- latent self-attention;
- sensor-back-attention;
- condition-context total;
- forward;
- backward;
- optimizer;
- whole step;
- peak allocated/reserved memory.

Use enough warmup and measured steps for stable results.

Where practical use `torch.profiler` to identify forward/backward attention kernels.

Also count K/V projection calls.

Expected qualitative behavior:

```text
legacy:
    K/V projected once per sensor-read/re-injection

cached_kv:
    K/V projected once per active sensor bucket
```

---

## 9. Small bucket comparison

Only test a few sensible bucket sets, for example:

```text
[288, 384]
[256, 320, 384]
one reasonable 4-bucket alternative
```

Choose based on whole-step time and memory.

Do not perform a broad sweep.

If bucketed execution is slower than cached-K/V/full padding, keep bucketing optional and promote only cached-K/V.

---

## 10. Short real-data smoke test

Do not launch 200/1000 epochs.

After correctness and benchmark gates pass, run only a short real-data smoke comparison:

```text
3–5 epochs
same seed
same B128/Q4096 protocol
same 2048 query microbatch
same optimizer/scheduler
```

Compare:

```text
frozen RC execution
best Stage-8 execution
```

Purpose:

- no NaNs/OOMs;
- normal loss behavior;
- correct EMA/checkpoint/resume;
- no obvious instability.

This is not a scientific accuracy study.

---

## 11. Promotion criteria

A Stage-8 execution mode is eligible for later long validation only if:

### Correctness

All attention/block/model/reconstruction parity tests pass.

### Efficiency

Minimum useful result:

```text
>= 8% lower whole B128/Q4096 training-step time
```

with no material memory regression.

Preferred:

```text
>= 12–15% faster whole step
and/or
>= 10% lower peak allocated memory
```

Do not require the coworker's reported 2x whole-training gain.

If only K/V caching is beneficial, promote only that.

---

## 12. Output

Store evidence under:

```text
_CheckNotes/Stage8_attention_optimization/
```

Include:

```text
README.md
correctness.json
benchmark.csv/json
bucket_comparison.csv/json
profiler/
smoke/
```

Final report should state:

- code/branch SHA;
- config switches;
- strict-load result;
- numerical/gradient/update parity;
- K/V projection-call reduction;
- whole-step and condition-attention speedup;
- memory change;
- best bucket policy;
- dynamic-trim diagnostic result;
- 3–5 epoch smoke result;
- recommendation for the later 1–2 long runs.

---

## Stage-8 decision rule

The Stage-8 change should remain an **exact execution optimization**:

> reuse condition-static attention work and avoid known padded work while preserving the frozen `GL_rbf_CQ` function and parameterization.

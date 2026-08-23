# Stage 6 — Compact Query Decoder

Status: all four gates complete. CQ-LR wins the prescribed comparison between
the two 128-wide candidates, but CQ does **not** replace F0 for formal 3-D work.
The single permitted CQ-160 rescue recovered some quality but remained
materially behind the immutable F0 best checkpoint.

## Scope and immutable reference

- Source branch verified at `169d7c545b9f980aed0fbaff0252e6d4114f3566`.
- Immutable F0: `F0_frozen_current_DemoN9300_20260821_075633/best.pt`.
- F0 SHA-256: `e93198bc2cba3f001024bbc9c1b197b2b56ecd52d8967bb38592ee5090e95569`.
- F0 was strictly loaded, never modified, and never retrained.

## Architecture and invariants

`GL_rbf_ENH_CQ` is a sibling of `GL_rbf_ENH`. It preserves the sensor encoder,
latent processor, latent reinjection, sensor back-attention/refinement,
`topk_rbf_glres`, K=32 KeOps semantics, learnable sigma, sensor importance,
Fourier coordinates, RFF prior, RF objective, Euler/Heun solvers, observation
consistency, and the Stage-1–5 data/execution path. Only query-side modules are
replaced.

| Model | Query width | Fusion/readout | Total params | Query params | Shared core |
|---|---:|---|---:|---:|---:|
| F0 | 256 | 640 concat + full block | 2,267,407 | 837,260 | 1,430,147 |
| CQ-Full | 128 | gated add + full 4-head block | 1,796,113 | 365,966 | 1,430,147 |
| CQ-LR | 128 | gated add + rank-64/4-head cached K/V | 1,643,665 | 213,518 | 1,430,147 |

The CQ-Full/LR common core has exactly the F0 parameter count. CQ-LR has no
per-query Transformer FFN and caches latent K/V in the existing condition
context. Both variants use the existing context, chunk, cache-level, streamed
reconstruction, and query-microbatch APIs.

## Gate 1 — implementation and invariants

Complete. F0 strict loading and the complete legacy regression suite remain
green: 78 tests passed, including 31 focused CQ tests. Tests cover shapes,
initialization, all preserved modules, gradient reachability, K=32/KeOps
semantics, configs, checkpoint metadata, and F0 compatibility.

Evidence: `implementation/gate_a_implementation.json`.

## Gate 2 — execution equivalence

Complete in FP32.

- Full versus cached-streamed reconstruction maximum absolute error:
  `1.1920929e-7` or lower across Euler NFE 1/2/4, Heun NFE 2, consistency modes,
  and static-feature caching.
- Monolithic versus 7-query microbatch loss difference: `1.1920929e-7`
  (CQ-Full), exactly zero (CQ-LR).
- Maximum gradient difference: `2.9802322e-8`.
- Maximum one-step AdamW update difference: `6.5052327e-6` (Full),
  `3.5180610e-6` (LR).
- Condition encoding runs once per cached trajectory; CQ-LR K/V projection also
  runs once, not once per NFE or query chunk.

Evidence: `equivalence/gate_b_equivalence.json`.

## Gate 3 — cost benchmark

Final idle-GPU measurements (milliseconds and peak allocated MiB):

| Queries | Model | Forward | Full train step | Peak MiB |
|---:|---|---:|---:|---:|
| 4,096 | F0 / Full / LR | 11.280 / 10.992 / 8.940 | 23.674 / 23.661 / 18.916 | 268.5 / 251.3 / 245.2 |
| 16,384 | F0 / Full / LR | 29.543 / 27.103 / 21.312 | 50.188 / 48.863 / 35.456 | 810.5 / 617.3 / 582.4 |
| 65,536 | F0 / Full / LR | 96.218 / 90.857 / 70.434 | 148.070 / 138.708 / 101.998 | 3,040.3 / 2,247.9 / 2,088.4 |

At 65,536 queries, CQ-LR is 1.366x faster forward and 1.452x faster per train
step than F0, with 31.3% lower peak allocation. It is 29.0% faster forward than
CQ-Full, exceeding the required 15% margin.

The component source of the gain is clear: at 65,536 queries, latent readout
drops from 15.239 ms (F0) / 12.581 ms (CQ-Full) to 5.192 ms (CQ-LR). The compact
point encoder drops from 1.240 to 0.689 ms, fusion/head from 2.058 to 0.847 ms,
and coarse head from 0.925 to 0.586 ms. The preserved local RBF gather remains
the dominant 42.106 ms bottleneck.

For cached one-million-query Euler NFE-2 reconstruction (three repetitions):

| Model | Mean wall s | s / 1M / NFE | Static cache MiB |
|---|---:|---:|---:|
| F0 | 0.4043 | 0.2021 | 2,197.3 |
| CQ-Full | 0.3882 | 0.1941 | 1,709.0 |
| CQ-LR | 0.3558 | 0.1779 | 1,709.0 |

CQ-LR is 13.6% faster than F0 for this end-to-end cached case and reduces the
static query cache by 22.2%. Peak total allocation is unchanged at 3,021 MiB
because CQ's smaller static cache leaves more measured dynamic headroom in the
same streamed execution envelope.

Evidence: `benchmarks/cost_benchmark.json`, `cost_benchmark.csv`, and
`gate_c_assessment.json`.

## Gate 4 — limited training screen

All screens use seed 42, batch 64, 4,096 monolithic queries, no query
microbatch, and the exact F0 optimizer/data/observation protocol. Setting
`scheduler_t_max: 200` preserves F0's learning-rate trajectory through epoch 60.

| Model | Mean epoch s (2–60) | Stored val @60 | Fixed RF @60 | Recon NFE 1/2/4 |
|---|---:|---:|---:|---|
| CQ-Full | 20.193 | 0.672566 | 0.618521 | 0.32375 / 0.35757 / 0.39820 |
| CQ-LR | 18.228 | 0.666365 | 0.625444 | 0.33553 / 0.36761 / 0.40712 |
| CQ-160 rescue | 22.412 | 0.654353 | 0.597152 | 0.31675 / 0.35438 / 0.39597 |
| F0 best (epoch 180) | — | 0.520273 best | 0.475357 | 0.29501 / 0.32503 / 0.36042 |

CQ-LR is 1.12% worse than CQ-Full on fixed RF and 2.85% worse on average
reconstruction, while adding 29.0% forward speed. It therefore passes all
three primary promotion criteria and is the primary CQ selection.

Both 128-wide variants were more than 5% behind F0 best in both controlled RF
and reconstruction, so exactly one allowed rescue was run: width 160, full
readout, with no other architecture/protocol changes. The rescue improved over
CQ-Full but still trails F0 best by 25.6% controlled RF and 8.84% average
reconstruction. Under the specification, the rescue therefore fails
materially and no further architecture family is opened.

Evidence is under `screen_cq_full/`, `screen_cq_lr/`,
`screen_cq_rescue160/`, and `formal_candidate/selection.json`.

## Formal candidate and recommendation

`formal_candidate/selected_200ep.yaml` prepares the rule-selected CQ-LR with
the exact F0 200-epoch protocol. `formal_candidate/launch.sh` requires explicit
`ALLOW_STAGE6_FORMAL_RUN=1`; it was tested only in refusal mode and the long run
was not launched.

**Recommendation: retain F0 for formal 3-D work.** CQ-LR is the best compact
query implementation and is worth keeping for memory- or throughput-constrained
inference, but the 60-epoch evidence does not justify replacing F0. The rescue
result indicates that the original query capacity remains important for formal
quality. A CQ-LR 200-epoch run should only be authorized as a deliberate
follow-up experiment, not treated as an already promoted formal model.

Decision figure and reproducibility contract:
`figures/generated/stage6_compact_query/`.

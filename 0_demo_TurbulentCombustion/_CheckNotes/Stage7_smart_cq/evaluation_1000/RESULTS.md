# Stage 7 final evaluation

## Decision

Recommend **Stage7-All256 (S7-B) as the new default CQ configuration**, using
`epoch_1000.pt` and EMA trainable weights for evaluation. It retains the fixed
128-D CQ-LR query decoder while improving quality and passing every efficiency
gate.

Primary configuration:

```yaml
backbone: GL_rbf_ENH_CQ
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
gather_topk: 32
```

Persistent Top-K, KeOps, sigma, GLRES, the RF objective, observations, data
protocol, and scheduler horizon remain unchanged.

## Controlled quality

Fixed-manifest protocol: 64 validation layouts x three repeats = 192 paired
rows, RF seed 1729, manifest checksum `392806184e0257f9...`.

| Candidate | checkpoint | RF mean | change vs F0 e1000 | paired loss-difference 95% CI |
|---|---:|---:|---:|---:|
| F0 | e1000 | 0.325531 | baseline | 0 |
| CQ-LR-128 | e1000 | 0.357043 | 9.7% worse | [0.02641, 0.03662] |
| CQ-LR-256 | best e840† | **0.261010** | 19.8% better | [-0.07203, -0.05701] |
| **Stage7-All256** | **e1000** | **0.261507** | **19.7% better** | **[-0.07159, -0.05646]** |

† The clean CQ-LR-256 run stopped at epoch 842; its best checkpoint is an
available partial-run reference, not a completed 1000-epoch control.

Stage7-All256 is 26.8% better than CQ-LR-128 at the exact epoch-1000 milestone
and reaches below F0's final RF mean by the first measured post-screen milestone,
epoch 400. It is statistically indistinguishable in practical magnitude from
the partial CQ-LR-256 reference (0.19% higher RF mean).

The stored best S7-B checkpoint is epoch 965. Its controlled RF mean is
0.261564, essentially tied with exact epoch 1000; exact epoch 1000 is slightly
better by 0.000057 and is recommended as the unambiguous scientific milestone.

## Matched reconstruction

One fixed validation snapshot, temperature conditioning, 256 shared sensors,
observation seed 42, RF seed 1729, Euler, and identical NFE1/2/4 randomness.
This is a deterministic diagnostic, not a dataset-wide uncertainty estimate.

| Candidate | NFE1 mean | NFE1 worst U1 | NFE4 mean | NFE4 worst U1 |
|---|---:|---:|---:|---:|
| F0 best e845 | 0.239674 | 0.568283 | 0.262493 | 0.657136 |
| CQ-LR-128 best e845 | 0.264192 | 0.607567 | 0.294014 | 0.707095 |
| CQ-LR-256 best e840† | **0.210119** | **0.438221** | **0.227882** | **0.525622** |
| **Stage7-All256 e1000** | **0.213053** | **0.471590** | **0.234270** | **0.557029** |

Versus F0, Stage7-All256 improves the five-field mean by 11.1% at NFE1 and
10.8% at NFE4; worst-field U1 improves by 17.0% and 15.2%, respectively. It is
1.4%/2.8% behind the incomplete CQ-LR-256 reference in NFE1/NFE4 mean.

All candidates now use the exact same frozen RF-prior checksum. The final audit
found that the original EMA implementation numerically averaged floating-point
buffers even when they were frozen; `prior.omega` drifted by at most 1.66e-4.
EMA now averages trainable parameters only and copies frozen parameters/buffers.
Existing Stage-7 checkpoints are repaired at load by taking frozen state from
their live checkpoint. The corrected fixed-manifest mean changed by only
1.2e-6, so the decision is unaffected.

## Formal efficiency/Pareto

Same RTX 6000 Ada, B128/Q4096, exact-gradient 2048-query execution microbatch
for Stage 7, and the existing formal benchmark protocol:

| Candidate | step (ms) | speedup | peak MiB | memory reduction | 1M/NFE4 (s) | inference speedup |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 544.84 | 1.00x | 27,346 | 0.0% | 0.4367 | 1.00x |
| CQ-LR-128 | 437.81 | 1.24x | 22,973 | 16.0% | **0.2433** | **1.79x** |
| **Stage7-All256** | **397.06** | **1.37x** | **20,239** | **26.0%** | **0.2857** | **1.53x** |

CQ-LR-128 remains the throughput-first choice when latency matters more than
quality. F0 is no longer Pareto-optimal under this protocol: Stage7-All256 is
both higher quality and cheaper. Stage7-All256 is the balanced/default choice.

## Separate kernel study

Artifact: `../benchmarks/attention_kernel_comparison.json`. Same selected
checkpoint, B128/Q4096, seven measured steps after two warmups.

- Six MHA modules and explicit SDPA have identical 5,490,617 parameters.
- Forward max absolute delta: 4.53e-6; loss delta: 4.77e-7; gradient max
  absolute delta: 9.54e-7; parity passed.
- MHA-mask + unfused AdamW: 403.85 ms.
- Explicit SDPA + unfused AdamW: 406.66 ms (**0.7% slower**), with 0.8% lower
  peak allocated memory. PyTorch 2.5 already dispatches the current
  `need_weights=False` MHA path efficiently, so keep MHA.
- MHA-mask + fused AdamW: 396.55 ms (**1.8% faster** overall); optimizer time
  drops from 8.35 to 1.71 ms. One-step parameter parity passes with a maximum
  difference of 1.49e-8.

Kernel recommendation: keep the historical MHA path. Fused AdamW is a valid
opt-in optimization, but its gain is small enough that it should remain separate
from the Stage-7 scientific default until longer-run optimizer stability is
confirmed.

## Verification and artifacts

- Complete suite after the EMA correction: **142 passed, 1 skipped**.
- `final_summary.json`: machine-readable recommendation.
- `final_comparison.csv`: quality, reconstruction, and formal cost table.
- `paired_statistics.csv`: paired RF differences and 95% CIs.
- `convergence.csv`: controlled milestone curves.
- `matched_reconstruction/summary.json`: deterministic field-level results.
- `../../../figures/generated/stage7_final_pareto/`: SVG, PDF, PNG, TIFF,
  and figure contract.

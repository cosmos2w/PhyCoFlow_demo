# Stage 7 — Smart CQ

Stage 7 keeps the frozen CQ-LR query decoder at 128 dimensions and adds optional condition/training improvements: model EMA, CQ-only sinusoidal time FiLM, and a raw per-field RBF value/support shortcut derived from the existing Top-K geometry.

## Provenance

- Frozen source commit: `6ac549b26a229222a209552bac344baeb86b7a4e`
- Annotated tag: `gl-rbf-enh-cq-v1`
- Remote tag object: `f60b5e5af95a360e5fa8cd2087fe3149385f778d`
- Stage-7 branch: `perf/pointcloud-smart-cq-stage7`

The tag was created and pushed before Stage-7 source changes. Existing performance branches and `main` were not moved.

## Candidate matrix

| Candidate | latent_dim | query decoder | EMA | time path | raw shortcut |
|---|---:|---|---|---|---|
| Frozen CQ-LR-128 | 128 | 128-D, LR64/H4, additive | off | scalar concat | none |
| S7-A / Stage7-Cond128 | 128 | 128-D, LR64/H4, additive | 0.999 | sinusoidal FiLM | RBF value/support |
| S7-B / Stage7-All256 | 256 | 128-D, LR64/H4, additive | 0.999 | sinusoidal FiLM | RBF value/support |

Both screens retain 128 latents, four latent blocks, K=32, KeOps, learnable sigma, GLRES, the clean optimized data path, 4096 effective queries, batch 128, seed 42, and scheduler horizon 1000. Training uses the tested exact-gradient 2048-query execution microbatch with a single reused condition context.

## Evidence map

- `implementation/CORRECTNESS.md`: implementation invariants and test evidence.
- `configs/`: validated 200-epoch screens and the selected S7-B continuation config.
- `evaluation_1000/RESULTS.md`: final controlled quality, reconstruction, Pareto, and default recommendation.
- `comparison/RESULTS.md`: compact Senseiver/latent-FM parameter, cost, NFE, memory, and reconstruction reference comparison.
- `benchmarks/`: pre-training cost/scaling and persistent-inference output.
- `benchmarks/attention_kernel_comparison.json`: separate MHA/SDPA/fused-AdamW parity and timing evidence.
- `screen_200/RESULTS.md`: controlled epoch-200 decision and caveats.
- `screen_200/evaluation/`: fixed-manifest, deterministic reconstruction, and consolidated comparison artifacts.
- `figures/generated/stage7_epoch200_pareto/`: editable comparison figure exports.
- `figures/generated/stage7_final_pareto/`: final quality/throughput recommendation figure.

Both formal screens completed, and S7-B / Stage7-All256 completed the sole
epoch-1000 continuation. It is the recommended default CQ configuration; retain
CQ-LR-128 as the throughput-first alternative.

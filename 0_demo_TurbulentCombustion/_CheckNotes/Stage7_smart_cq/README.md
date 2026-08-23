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

Both screens retain 128 latents, four latent blocks, K=32, KeOps, learnable sigma, GLRES, the clean optimized data path, 4096 queries, batch 128, seed 42, and scheduler horizon 1000.

## Evidence map

- `implementation/CORRECTNESS.md`: implementation invariants and test evidence.
- `configs/`: validated 200-epoch S7-A/S7-B configs.
- `benchmarks/`: pre-training cost/scaling and persistent-inference output.
- `screen_200/`: formal 200-epoch run artifacts.
- `comparison/`: fixed-manifest, reconstruction, and Pareto comparison.

Long scientific runs must not start until the pre-training efficiency gates pass.

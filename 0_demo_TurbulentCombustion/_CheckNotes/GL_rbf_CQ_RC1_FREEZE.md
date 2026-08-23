# GL_rbf_CQ v0.9.0-rc1 freeze record

- Public release-candidate name: **`GL_rbf_CQ`**
- Annotated tag: **`gl-rbf-cq-v0.9.0-rc1`**
- Tag object SHA: `434eabc28fbd9a17a822ae6b467b476916afb6e3`
- Frozen source/documentation commit: `1b9a6d47f6c248364df6ba54155b5eac3d5e6e67`
- Branch: `perf/pointcloud-smart-cq-stage7`
- Repository: `cosmos2w/PhyCoFlow_demo`
- Freeze date: 2026-08-23 (America/New_York)
- Verification: **143 passed** in the full PointCloud test suite on physical GPU 1

The tag is the immutable validated Stage 1–7 RC snapshot. This freeze note is a
post-tag record, so its documentation-only commit is intentionally one commit
after the tag target.

## Selected research artifact

- Exact milestone: `ReleaseArtifacts/GL_rbf_CQ_rc1/GL_rbf_CQ_v0.9.0-rc1_e1000_research.pt`
- SHA-256: `31e59110258d4cc4715e13a5c92efb01d1eec72bdac7da3a3cec384da6f2042a`
- Selection: epoch 1000, EMA trainable weights plus exact live frozen state
- Companion best checkpoint: epoch 965, SHA-256
  `e4c97bcb6385b7ec666baff652009068bfba2fa473c472f8b49469bdc40d7fc9`
- Complete binary inventory: `_CheckNotes/GL_rbf_CQ_rc1_artifacts.md`

The checkpoint files are ignored by Git and are recorded by stable local path
and checksum. They are research checkpoints, not yet portable single-state
deployment exports.

## Frozen balanced/default configuration

```yaml
backbone: GL_rbf_ENH_CQ  # frozen internal identifier; public name GL_rbf_CQ
latent_dim: 256
num_latents: 128
num_latent_blocks: 4
cq_query_dim: 128
cq_readout_mode: lowrank
cq_readout_rank: 64
cq_readout_heads: 4
cq_fusion_mode: additive
cq_time_conditioning: sinusoidal_film
cq_time_embed_dim: 128
cq_time_film_zero_init: true
cq_measurement_support_mode: rbf_value_support
cq_measurement_support_normalize: true
model_ema_enabled: true
model_ema_decay: 0.999
model_ema_eval: true
gather_mode: topk_rbf_glres
gather_topk: 32
neighbor_backend: keops
```

Persistent Top-K geometry, learned sigma, GLRES sensor importance, random-field
objective, optimized data path, query microbatching, and cached-streamed
reconstruction are part of the frozen behavior.

## Public profiles

- **`GL_rbf_CQ`**: recommended balanced/default profile above.
- **`GL_rbf_CQ-fast`**: frozen CQ-LR-128 profile with persistent Top-K; fastest
  measured persistent inference, with a known 9.7% controlled RF penalty versus
  F0 at epoch 1000.
- **`GL_rbf_ENH`**: legacy/reference profile retained for reproducibility.

## Validation summary

At B128/Q4096 on the matched RTX 6000 Ada protocol, `GL_rbf_CQ` gives
`397.06 ms` per step, `20,239 MiB` peak allocation, and `0.2857 s` persistent
1M-query Euler NFE-4 inference. Its epoch-1000 fixed-manifest RF mean is
`0.261507`, versus `0.325531` for F0.

The complete technical record is `ModelUpdate.md`; the canonical evidence and
worktree classification are in `_CheckNotes/GL_rbf_CQ_RC1_WORKTREE_AUDIT.md`.

## Cleanup status

**Cleanup has not been performed.** No runtime source was refactored, no model
or directory was renamed, no worktree was removed, and no run/checkpoint was
deleted in the RC freeze. `Stage7_Clean_Up.md` is a future, phased plan whose
compatibility oracle is this tag.

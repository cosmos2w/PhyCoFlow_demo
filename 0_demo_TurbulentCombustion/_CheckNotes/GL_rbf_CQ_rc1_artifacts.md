# GL_rbf_CQ v0.9.0-rc1 artifact manifest

This manifest records the local research artifacts associated with the frozen
`GL_rbf_CQ` RC1 documentation. Model files are intentionally excluded from Git
by the repository-wide `*.pt` rule; the stable paths and hashes below are the
canonical local inventory. Copy by checksum, not by filename alone.

## Source lineage

- Frozen branch: `perf/pointcloud-smart-cq-stage7`
- Validated branch tip before RC documentation: `1a75d3a4f449dc6fdc6123bdff6d7ae9e7c3aca0`
- Stage 7 implementation: `07bc220`
- Stage 7 efficiency/correctness revision used by the run: `6d996819ef20f2b03a43e4f322b02891c099fcfb`
- Final EMA loading/evaluation revision: `a03213ae33f14cbffd198c2da65aa8a11e6b2f26`
- The research checkpoint format does not embed a Git SHA. Therefore
  `6d996819...` is the recorded training-source lineage, while `a03213ae...`
  is the exact corrected loader/evaluator lineage. This distinction is
  deliberate rather than presenting an inferred SHA as checkpoint metadata.

## Stable local artifacts

Base directory: `ReleaseArtifacts/GL_rbf_CQ_rc1/`

| Purpose | Stable path | Bytes | SHA-256 | Epoch / weights | Notes |
|---|---|---:|---|---|---|
| Exact scientific milestone | `GL_rbf_CQ_v0.9.0-rc1_e1000_research.pt` | 88,086,043 | `31e59110258d4cc4715e13a5c92efb01d1eec72bdac7da3a3cec384da6f2042a` | epoch 1000; live + EMA; EMA selected | Recommended research checkpoint; 71,000 EMA updates; corrected loading combines EMA trainable tensors with exact live frozen state. |
| Best-validation companion | `GL_rbf_CQ_v0.9.0-rc1_best_e965_research.pt` | 88,047,445 | `e4c97bcb6385b7ec666baff652009068bfba2fa473c472f8b49469bdc40d7fc9` | epoch 965; live + EMA; EMA selected | Validation loss 0.292274; controlled fixed-manifest quality is practically tied with epoch 1000. |
| Exact training configuration | `run_config_training.yaml` | 3,028 | `11685bc3662e99177ba20f1aed5d518d6631ab4f72139ed38e6494ec978e7ca5` | n/a | Preserves the historical absolute paths from the run; use the public preset in `ModelUpdate.md` for portable configuration. |
| Normalization statistics | `dataset_stats.pt` | 1,468 | `a3f3efb8a552af5804315e15ea21afb871585c88574f81e9a4ea4b59ee3f999a` | n/a | Dictionary with `mean` and `std`; copied from the stable DemoN51 statistics file. |

## Copy verification

The checkpoint files were copied from the detached Stage 7 evidence worktree
into the stable release directory. Source and destination SHA-256 digests were
identical for both checkpoints. The source run was:

`_CheckNotes/Stage7_smart_cq/screen_200/runs/S7_B_All256_200ep_B128_DemoN9702_20260822_224830/`

The source path is retained here for provenance only. Consumers should use the
stable `ReleaseArtifacts/GL_rbf_CQ_rc1/` paths.

## Checkpoint semantics

The two research checkpoints contain 148 live model tensors and 148 EMA shadow
tensors. EMA decay is `0.999`; the exact-milestone checkpoint has 71,000 EMA
updates and the best checkpoint has 68,515. Historical Stage 7 EMA shadows
averaged floating-point frozen buffers. The RC loader repairs this by selecting
EMA values only for trainable state and copying frozen parameters/buffers from
the live checkpoint. All matched candidates then share RF-prior checksum:

`bd2a5d9eb7cc1339b97648048110650a0121ba874fc13da87c6917cecd32cd1a`

These remain research checkpoints, not coworker-portable EMA-resolved release
exports. Producing a single resolved deployment checkpoint belongs to the
post-freeze cleanup plan and is not performed in this RC documentation pass.

## Reference checkpoints retained in existing stable directories

| Public/reference role | Checkpoint | Scope |
|---|---|---|
| Legacy quality baseline `GL_rbf_ENH` | `_CheckNotes/Stage6_clean_ab/runs/F0_ENH_1K_B128_DemoN9510_20260821_235104/epoch_1000.pt` and `best.pt` | Completed 1000-epoch F0 run; live weights. |
| Throughput preset source `GL_rbf_CQ-fast` | `_CheckNotes/Stage6_clean_ab/runs/CQ_LR_1K_B128_DemoN9511_20260821_235104/epoch_1000.pt` and `best.pt` | Completed CQ-LR-128 run; historical features disabled. |
| High-capacity clean control | `_CheckNotes/Stage6_clean_ab/runs/CQ_LR_L256_1K_B128_DemoN9561_20260822_144624/best.pt` | Partial run, best epoch 840; reference only, not a completed RC candidate. |

No run directories or checkpoints were deleted during this freeze.

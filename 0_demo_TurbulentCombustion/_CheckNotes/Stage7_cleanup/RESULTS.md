# Stage-7 post-RC cleanup results (Phases 0–5)

## Outcome

The normal checkout now provides a self-contained public `GL_rbf_CQ` interface
while preserving `GL_rbf_ENH_CQ`, checkpoint keys, model mathematics, RNG order,
EMA repair, RF objective, persistent Top-K, and query-microbatch semantics.

## Compatibility evidence

- Pre-cleanup oracle suite: 143 passed.
- Post-cleanup full suite: 154 passed.
- The deterministic three-checkpoint oracle JSON is byte-identical before and
  after the factory/checkpoint refactor (SHA256
  `56484012df5d4208b106fb30fa8eecaca0b24b5284eaf19677d1bdac4f1ad395`).
- Balanced resolved state: 148 keys, tensor SHA256
  `f1c92d4bcf1b9e0ac90ad20b3b3468764f6cca200dde1575b914d9ab68d7b99f`.
- Fixed-manifest public-checkpoint evaluation: 192 rows, RF mean
  `0.2615070373285562`, manifest SHA256 `392806184e0257f9...`, materialized-input
  SHA256 `b116f09b4ffc40bf...`; all exactly match the RC1 e1000 result.
- Real-data public reconstruction command completed with persistent
  static-feature caching and Euler NFE-1.

## Performance sanity

Same RTX 6000 Ada protocol, compared with the recorded RC1 Stage7-All256 row:

| Protocol | Cleanup / RC1 time | Cleanup / RC1 peak allocation |
|---|---:|---:|
| B128/Q4096 training step, microbatch 2048, EMA | 0.9603 | 0.9979 |
| 1M-query persistent Euler NFE-4 | 0.9919 | 0.9813 |

There is no material regression. Full raw measurements are in
`phase5/performance_sanity.json`.

## Artifact

- Portable checkpoint:
  `ReleaseArtifacts/GL_rbf_CQ_rc1/GL_rbf_CQ_v0.9.0-rc1_e1000_ema_resolved_portable.pt`
- File SHA256:
  `2516ffeb45775d4e6b8d88b4b24d927aac28665a2a90102583e07deaca78f64d`
- Size: 22,023,920 bytes
- Weights: corrected legacy EMA trainable shadow plus live frozen state.

## Scope retained for Phase 6

No permanent deletion or linked-worktree removal was performed. See
`WORKTREE_AUDIT.md` and `research/MAP.csv` for reviewed candidates.

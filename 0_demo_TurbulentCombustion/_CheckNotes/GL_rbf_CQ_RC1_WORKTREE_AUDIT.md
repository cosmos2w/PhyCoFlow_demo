# GL_rbf_CQ RC1 worktree and evidence audit

Audit date: 2026-08-23 (America/New_York)

## Canonical checkout

The normal repository checkout at `/home/wanglz/Desktop/src/PhyCoFlow` is the
canonical RC working copy. It is on `perf/pointcloud-smart-cq-stage7`, tracks
`origin/perf/pointcloud-smart-cq-stage7`, and was fast-forward synchronized to
`1a75d3a4f449dc6fdc6123bdff6d7ae9e7c3aca0` before RC documentation began.

The remote had advanced from the earlier evaluated `a03213a` by one commit,
`1a75d3a docs(pointcloud): add Stage 7 baseline comparison`. Its diff contains
only Stage 7 comparison reports/scripts/data and no scientific model-source
change, so `1a75d3a` is the validated documentation tip used for the freeze.

## Worktree classification

| Worktree | Commit / branch | Classification | RC action |
|---|---|---|---|
| normal checkout | `1a75d3a`, `perf/pointcloud-smart-cq-stage7` | canonical source and tracked evidence | Promoted to the normal checkout; all new RC documentation is created here. |
| `.worktrees/pointcloud-smart-cq-stage7` | detached `1a75d3a` | Stage 7 run artifacts, checkpoints, logs, reconstruction plates | Canonical tracked summaries already match normal checkout. Selected e1000/e965 checkpoints copied to stable release paths with verified hashes. Worktree retained. |
| `.worktrees/pointcloud-cq-balanced` | `beaa94d`, `perf/pointcloud-cq-balanced` | rejected structured-concat cost-gate experiment | Canonical negative-result report and benchmark evidence already exist in normal checkout. No unique accepted source. Worktree retained. |
| `/tmp/phycoflow_cq_no_topk_01d2847` | detached `01d2847` | historical non-persistent training A/B run material | Canonical README/summary/benchmark evidence exists in normal checkout; large run products are research logs. Retained, not promoted. |
| `/tmp/phycoflow_cq_topk_3f3eefb` | detached `3f3eefb` | historical persistent-Top-K training A/B run material | Canonical README/summary/benchmark evidence exists in normal checkout; large run products are research logs. Retained, not promoted. |

Checksum-aware dry-run comparisons were used so timestamp-only differences were
not mistaken for unique content. No model implementation existed only in an
unmerged worktree. TIFFs missing from the normal checkout were regenerated as
part of the RC figure bundle from canonical source data rather than copied
blindly.

## Artifact disposition

- **Canonical tracked evidence:** Stage 1–7 reports, evaluation JSON/CSV,
  benchmark scripts, figure scripts, SVG/PDF/PNG figures, and this audit.
- **Stable local binary evidence:** selected Stage 7 checkpoints, exact run
  YAML, and dataset statistics in `ReleaseArtifacts/GL_rbf_CQ_rc1/`.
- **Research-only local material:** milestone checkpoints, terminal logs,
  diagnostics streams, temporary reconstruction products, and pycache files.
- **Preserved inputs:** `_CheckNotes/Codex_Stage7_Smart_CQ_GoalMode.md` and
  `_CheckNotes/GL_rbf_CQ_RC_CleanupPlan.md` are included as provenance rather
  than discarded as unexplained untracked files.

## Non-actions

No worktree was removed, no source was refactored, no directory was renamed,
and no checkpoint/run artifact was deleted. Those actions would be cleanup, and
this RC pass is explicitly freeze/documentation only.

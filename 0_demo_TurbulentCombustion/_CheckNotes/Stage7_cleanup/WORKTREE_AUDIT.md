# Linked-worktree audit

No linked worktree was removed in Phases 0–5. The normal checkout has no source,
config, test, script, or artifact-manifest reference to `.worktrees/`; only the
historical cleanup-plan size table mentions that directory.

| Worktree | Size | Finding | Decision |
|---|---:|---|---|
| `.worktrees/pointcloud-cq-balanced` | 42 MiB | clean tracked tree; only ignored pytest/Python caches; branch `perf/pointcloud-cq-balanced` preserves its commit | retain until Phase-6 review |
| `.worktrees/pointcloud-smart-cq-stage7` | 2.0 GiB | unique ignored checkpoints, matched-reconstruction images, logs, and run backups remain | retain; removal would lose local evidence |
| `/tmp/phycoflow_cq_no_topk_01d2847` | 96 MiB | unique ignored no-persistent training checkpoint/log artifacts remain | retain; archive artifacts first |
| `/tmp/phycoflow_cq_topk_3f3eefb` | 96 MiB | unique ignored persistent-Top-K training checkpoint/log artifacts remain | retain; archive artifacts first |

The audit used `git worktree list --porcelain` and `git -C <worktree> status
--short --ignored`. Any future removal must use `git worktree remove` after a
fresh artifact/hash comparison; manual directory deletion is prohibited.


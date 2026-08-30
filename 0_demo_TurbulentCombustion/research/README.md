# Research provenance index

The collaborator-facing training runtime is `src/train_pointcloud_ffm.py`, with
reusable code in `src/phycoflow_pointcloud/` and public profiles in `configs/`.
Stage-specific reports and exact-run configurations remain in
their original tracked `_CheckNotes/` paths for link stability and tag-level
recovery. They are indexed by `MAP.csv` rather than moved in the same revision
that introduces their replacements.

Completed Stage 2–7 benchmark programs are archived under
`research/benchmarks/`; the Stage-7 decision analyzer is under
`research/stages/stage7/`. They remain runnable and tested, but are not part of
the normal collaborator-facing source surface.

Compatibility entry points retained:

- `src/train_pointcloud_ffm.py`
- `src/reconstruct_pointcloud.py`
- `src/evaluate_fixed_manifest.py`
- `src/evaluate_pointcloud_fixed_manifest.py`
- `research/benchmarks/benchmark_pointcloud_reconstruction.py`
- `Model.ConditionalPointHybridLocalGlobalRBFCQ`
- YAML `backbone: GL_rbf_ENH_CQ`

## Durable decisions

| Decision | Status | Evidence/reason |
|---|---|---|
| Latent-256 scientific architecture as `GL_rbf_CQ` | accepted | best fixed-manifest/reconstruction Pareto at epoch 1000 |
| Latent-128 CQ-LR as `GL_rbf_CQ-fast` | accepted | lowest validated CQ cost profile |
| Persistent geometry-only Top-K | accepted | equivalent outputs and zero post-build searches |
| Cached K/V with full padding | accepted execution default | matched mature RF/reconstruction; 6.1–6.4% faster, 2.41% lower peak allocation |
| Static bucketing and dynamic trimming | rejected as defaults | neither improves on cached/full for the release workload |
| Structured-concat 192/224 sweep | rejected as default | did not justify replacing the validated additive CQ path |
| SDPA/fused optimizer in cleanup | excluded | kernel/optimizer changes are outside behavior-preserving RC cleanup |

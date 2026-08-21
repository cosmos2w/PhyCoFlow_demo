# Stage 6 current-architecture formal baseline

This package prepares two 200-epoch runs against the frozen Stage 1–5 model at
`169d7c545b9f980aed0fbaff0252e6d4114f3566`
(`stage1-5-optimized-reference-v1`). It does not contain Stage 6 architecture
changes and does not launch training automatically.

## Matched protocol

Both runs use seed 42, batch size 144, the same dataset split, Adam settings,
GL_rbf_ENH widths/blocks, `topk_rbf_glres`, K=32, RFF prior and Rectified-Flow
objective, optimized Stage 1–5 data handling, and cached-streamed
reconstruction. Batch 144 is the active-config value; the earlier limited runs
used 96 only to accommodate a co-tenant process.

- F0 (`Demo_Num: 9300`): 4,096 effective queries, monolithic execution.
- F1 (`Demo_Num: 9301`): 16,384 effective queries, 8,192-query execution
  microbatches, with condition context reused.
- Validation runs every 5 epochs. Final cached-streamed reconstruction runs at
  epoch 200 for Euler NFE 1, 2, and 4.
- Generated logs, run directories, and initial GPU snapshots are ignored by
  Git. Checkpoints remain local until explicitly selected for publication.

The validator guarantees that the two YAML files differ only in `Demo_Num`,
`save_dir`, `n_query_points`, and `train_query_microbatch_size`; it also checks
the architecture/RF/reconstruction keys against the active main config.

## Launch commands

Run from the repository root. The commands below select physical GPU 1; inside
the process it is remapped to logical `cuda:0`, matching `device_ids: [0]`.

First validate without training:

```bash
conda run --no-capture-output -n phycoflow_env \
  python _CheckNotes/Stage6_formal_baseline/validate_configs.py
```

Then launch F0 and F1 separately (do not overlap them on the same GPU):

```bash
GPU_ID=1 bash _CheckNotes/Stage6_formal_baseline/launch.sh F0
```

```bash
GPU_ID=1 bash _CheckNotes/Stage6_formal_baseline/launch.sh F1
```

The launcher records initial GPU state and refuses to start if tracked training
or model source has diverged from the frozen reference. Confirm GPU 1 is empty
before each launch. F1 should be watched closely during its first optimizer
step because its 8,192-query execution chunk is larger than the prior limited
16k/4k integration run.

## Decision after evaluation

- If F1 is clearly no better, retain F0 as the formal baseline.
- If F1 materially improves reconstruction or convergence, replicate F1 only
  with two additional seeds for formal statistics.

Post-processing, fixed-manifest evaluation, reconstruction comparison, and the
two additional seed configs are intentionally deferred until the two runs have
finished and been reviewed.

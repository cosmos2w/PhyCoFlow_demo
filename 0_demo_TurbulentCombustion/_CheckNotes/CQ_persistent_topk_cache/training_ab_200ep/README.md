# CQ-LR persistent Top-K — 200-epoch code A/B

This experiment compares the same CQ-LR model using two pinned source revisions:

- **without persistent Top-K:** `01d2847`, immediately before the feature;
- **with persistent Top-K:** `3f3eefb`, the implemented and benchmarked feature.

The training/data-path files are byte-identical between those revisions. Both
jobs use seed 42, batch 128, 4,096 monolithic query points, the same observations,
optimizer, 200-epoch cosine schedule, data path, and CQ-LR architecture. YAMLs
differ only in `Demo_Num` and `save_dir`.

Persistent Top-K is an inference-only facility. Therefore the two training runs
are a quality-neutrality control, not the primary speed benchmark. After both
runs finish, the launcher automatically:

1. evaluates milestones 1/20/40/60/100/150/200 on the fixed validation manifest;
2. benchmarks both epoch-200 checkpoints at 250k and 1M queries for Euler
   NFE 1/2/4/8, including prior per-call Stage-4 and persistent cache modes;
3. checks checkpoint weights, loss curves, output equivalence, KNN calls,
   timing, and memory;
4. writes `comparison.json`, `RESULTS.md`, and SVG/PDF/PNG figures.

The per-run `loss_history.png` files refresh every five epochs. The final figure
is written under `figures/generated/cq_persistent_training_ab/`.

## Launch

```bash
OLD_GPU=0 NEW_GPU=1 bash _CheckNotes/CQ_persistent_topk_cache/training_ab_200ep/launch_pair.sh
```

The launcher refuses busy GPUs, creates detached pinned worktrees under `/tmp`,
and records the controller and worker information in `active_pair.env` and
`pipeline_status.env`.

Monitor with:

```bash
cat _CheckNotes/CQ_persistent_topk_cache/training_ab_200ep/pipeline_status.env
tail -f _CheckNotes/CQ_persistent_topk_cache/training_ab_200ep/logs/CQ_LR_*.log
nvidia-smi
```

## Interpretation

Training time and memory should remain statistically neutral because neither
training nor validation loss invokes persistent reconstruction geometry. The
relevant speed comparison is the previous `static_features` per-call path versus
`static_persistent_geometry` during repeated reconstruction. Quality acceptance
requires matched fixed-manifest RF and validation loss within 0.5%, output
differences at most `1e-5`, zero post-build KNN calls, and at least 15% steady
reconstruction speed improvement.

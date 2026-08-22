# Figure contract

## Core scientific claim

Persistent geometry-only Top-K reuse reduces repeated CQ-LR reconstruction
latency while leaving the training trajectory and fixed-manifest RF quality
unchanged within the predeclared tolerance.

## Source files

- `/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/CQ_persistent_topk_cache/training_ab_200ep/comparison.json`
- both run directories' `loss_history.json`
- `benchmarks/no_persistent_checkpoint.json`
- `benchmarks/persistent_topk_checkpoint.json`

## Panel map

- a: paired 200-epoch training RF curves;
- b: paired validation RF curves;
- c: one-million-query steady latency at Euler NFE 1/2/4/8;
- d: persistent speedup over the prior per-call Stage-4 static cache.

## Metrics and caveats

Latency is CUDA-synchronized and excludes the separately reported one-time
geometry build. Training timing is a neutrality control because persistent
Top-K is not used in the RF training objective. This is a one-seed, two-GPU
paired implementation check rather than a multi-seed scientific comparison.

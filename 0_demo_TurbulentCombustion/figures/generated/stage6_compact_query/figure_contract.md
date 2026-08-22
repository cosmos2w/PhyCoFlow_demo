# Figure contract — Stage 6 compact query decoder

## Core scientific claim

The compact query decoder reduces repeated-query execution and memory while
the selected 60-epoch candidate remains within the prespecified CQ quality
screen. Primary selected candidate: CQ-LR. Status: prepared_not_recommended_rescue_failed.

## Source files

- /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_compact_query/benchmarks/cost_benchmark.json
- /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_compact_query/screen_cq_full/evaluation/fixed_manifest/milestones.json
- /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_compact_query/screen_cq_lr/evaluation/fixed_manifest/milestones.json
- /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_compact_query/screen_cq_full/evaluation/matched_reconstruction/epoch_0060/summary.json
- /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_compact_query/screen_cq_lr/evaluation/matched_reconstruction/epoch_0060/summary.json
- /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_formal_baseline/evaluation/fixed_manifest_best.json
- /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_formal_baseline/evaluation/matched_reconstruction/F0_best/summary.json
- /home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion/_CheckNotes/Stage6_compact_query/formal_candidate/selection.json

## Panel map

- a: measured model forward scaling at 4,096/16,384/65,536 queries.
- b: measured 65,536-query training peak and 1M static cache.
- c: 64-layout, three-repeat fixed-manifest RF loss at epochs 1/20/40/60.
- d: matched snapshot/sensors/RF seed Euler NFE 1/2/4 reconstruction.

## Metrics/statistics

Timings and memory are same-run CUDA measurements. RF points are means over
the fixed 64 layouts and three RF repeats per layout. Reconstruction is the
five-field mean relative L2 on the established controlled snapshot.

## Caveats

The CQ screen is one seed and only 60 epochs. F0 is its 200-epoch best
checkpoint, so panel c/d do not establish formal CQ-versus-F0 parity. That
requires the prepared selected-candidate 200-epoch run.

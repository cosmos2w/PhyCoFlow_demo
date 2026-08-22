# Clean F0-ENH versus CQ-LR A/B

This package measures the Stage-6 query-decoder change without mixing in older
datasets, objectives, data paths, supervision counts, or training schedules.

## Compared models

- **F0-ENH:** `GL_rbf_ENH`, the pre-Stage-6 formal architecture.
- **CQ-LR:** `GL_rbf_ENH_CQ`, query width 128 with rank-64, four-head cached
  latent readout.

The YAML files differ only in `backbone`, `Demo_Num`, and `save_dir`. All model
core, data, RF, optimizer, observation, and execution settings are identical.
Both use seed 42, batch 64, 4,096 monolithic queries, a 200-epoch cosine horizon,
and train for a 60-epoch screen.

## Launch both GPUs

~~~bash
OLD_GPU=0 NEW_GPU=1 bash _CheckNotes/Stage6_clean_ab/launch_pair.sh
~~~

The launcher refuses busy GPUs, runs both jobs concurrently, evaluates
checkpoints 1/20/40/60 on the fixed 64-layout manifest with three RF repeats,
and writes `comparison.json` plus `RESULTS.md`.

Monitor while running:

~~~bash
tail -f _CheckNotes/Stage6_clean_ab/logs/F0_ENH_*.log \
        _CheckNotes/Stage6_clean_ab/logs/CQ_LR_*.log
nvidia-smi
~~~

## Reported metrics

- mean/median epoch time and total recorded wall time;
- time and epoch required to cross training-loss thresholds 1.0/0.8/0.7;
- validation curve, best/final validation, and controlled fixed-manifest RF;
- sampled training-step time;
- maximum process-local CUDA allocated and reserved memory.

This is a one-seed screen. It cleanly measures the paired implementation
behavior, but training-seed uncertainty requires later replication.

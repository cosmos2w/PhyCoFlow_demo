# Figure 5 V4 validation workflow

## V4.1 additive revision

V4.1 preserves the complete V4 bundle and adds distribution-aware a/b panels,
log–log c/d planes, a two-GPU canonical Geo-FNO training-memory replay, a taller
memory-only e panel, tighter gutters, larger typography/legend, and a separate
audited Zero-H-balanced four-panel backup. Its source/statistical contract is
`docs/figure5_v41_source_schema.md`.

The Geo-FNO DDP run promotes only process-local allocated memory; wall timing
under shared GPU load is explicitly inadmissible:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc-per-node=2 \
  Dis_SI_Process/scripts/benchmark_geofno_ddp_v41.py \
  --execute --confirm-in-memory-replay --memory-only \
  --run-id geofno_ddp_memory_formal_v41
```

Then build all five V4.1 main standalones, the 183-mm composed main figure,
four Zero-H backup standalones, the backup composite, source tables,
companions, manifest, completion report, and SVG QA:

```bash
conda run -n fig python Dis_SI_Process/scripts/build_figure5_v41.py \
  --strict-formal --timestamp YYYYMMDD_HHMM
```

The strict build rejects a missing/non-passing two-GPU run, a non-formal
inherited V3/V4 input, any bootstrap mismatch, or an unaudited Zero-H source.

This additive workflow builds Figure 5 V4 while preserving every V2/V3 result
as provenance. Panels a--c reuse the QA-passing V3 UQ and native clean-cost
runs in place. Panels d/e use independent `ValidationV4` roots and can never
fall back to V2 timing, a V3 query proxy, file timestamps, or fabricated query
scaling.

Under `Cond_T`, `Y_CH4`, `Y_CO`, `U1`, and `p` are macro-aggregated with equal
0.25 weights where applicable.

## V4 panel map

- `a` -- normalized empirical CRPS across the five trained generative methods.
- `b` -- method-wise spread/error Spearman association, described as
  uncertainty informativeness rather than calibration.
- `c` -- validated native N=40,300 inference accuracy--latency for all eight
  Figure 4 checkpoints.
- `d` -- reconstruction error versus directly measured clean-GPU training
  update time at each method's adopted canonical batch/query configuration.
- `e` -- full-width latency/memory scalability envelope. Only DMF-Gen,
  FFM-Perceiver, MLP-RBF, and Senseiver receive arbitrary-query curves;
  FFM-FNO, Latent FM, SiT, and adopted Geo-FNO are native-only markers.

Full reliability/interval-width curves, fieldwise UQ, diversity, cold/no-cache
timing, reserved memory, stage-level training replay, and NFE/solver
diagnostics remain SI/internal. No ablation training is part of V4.

## Adopted formal sources

- V3 UQ: `uq_compare_formal_20260830_v3r6`
- V3 native inference: `formal_cost_clean_v3_20260830_v3`
- V4 historical training audit: `training_cost_formal_v4` (correctly blocked;
  historical GPU-hours are incomplete/incomparable)
- V4 direct training replay: `training_replay_formal_v4r2`
- V4 scale stress: `scale_stress_formal_v4`

The V2 approximately 127-ms DMF latency remains superseded by the V3 clean
warm model-core coordinate and is not a V4 fallback.

## Training-cost audit and replay

Use `phycoflow_env`. The audit reads explicit checkpoint/update metadata but
never uses filesystem modification times:

```bash
conda run -n phycoflow_env python \
  Dis_SI_Process/scripts/audit_training_cost_v4.py \
  --config Dis_SI_Process/configs/training_cost_audit_v4.yaml \
  --run-id training_cost_formal_v4 --strict
```

The expected nonzero strict result records that historical GPU-hours cannot be
promoted. The direct fallback is an in-memory canonical training-update
benchmark using every stage's adopted batch/query configuration. It performs
20 warmups and 100 measured updates for each successful stage, attempts all
nine required stages, writes no checkpoint/history, and verifies archived
hashes remain unchanged:

```bash
conda run -n phycoflow_env python \
  Dis_SI_Process/scripts/benchmark_training_replay_v4.py \
  --execute --confirm-in-memory-replay --device cuda:2 \
  --run-id training_replay_formal_v4r2
```

Six methods are admissible in panel d: DMF-Gen, FFM-FNO, FFM-Perceiver, SiT,
MLP-RBF, and Senseiver. Latent FM's shared autoencoder and flow stages are both
measured, but no single method-level update-time coordinate is invented; their
stage values remain SI. Geo-FNO's adopted batch of 192 exceeded the 47.38-GiB
device capacity, so its method-level coordinate is also explicitly unavailable.

## High-N scale stress

Run only on a clean GPU. The runner uses one sensor-prefixed deterministic
Sobol specification and predeclared 100k, 250k, 500k, 1M, 2M, and 4M points,
then adaptively attempts 8M. Values above 40,300 are throughput-only and carry
no accuracy claim.

```bash
conda run -n phycoflow_env python \
  0_demo_TurbulentCombustion/tools/benchmark_validation_v4_scale_stress.py \
  --plan 0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_ValidationPlans/validation_v1.yaml \
  --device cuda:2 --run-id scale_stress_formal_v4 \
  --global-cap 8000000 --warmups 20 --minimum-repeats 30 \
  --minimum-seconds 10 --runtime-cap-seconds 60 \
  --memory-fraction 0.90 --query-chunk-size 8192
```

The formal capacity results are: DMF-Gen reaches the 8M cap;
FFM-Perceiver succeeds at 4M and fails at 8M; MLP-RBF and Senseiver succeed at
1M and fail at 2M.

## Strict figure build

Use the plotting-only `fig` environment:

```bash
conda run -n fig python Dis_SI_Process/scripts/build_figure5_v4.py \
  --strict-formal --timestamp <YYYYMMDD_HHMM> \
  --preview-png /tmp/figure5_v4_preview.png
```

The build emits five timestamped standalone editable SVGs, one compact 183-mm
composed SVG, one companion Markdown per panel plus a composed companion and
completion report, and timestamp-matched derived source tables/manifests/QA.
Strict mode fails if panel d/e evidence is absent or unsupported.

## QA

```bash
conda run -n fig python Dis_SI_Process/scripts/audit_figure5_v4.py \
  Dis_SI_Process/figures/generated/<timestamp> --strict-formal

conda run -n fig python -m unittest \
  Dis_SI_Process.tests.test_figure5_pipeline \
  Dis_SI_Process.tests.test_figure5_v4_pipeline

conda run -n phycoflow_env python -m pytest -q \
  Dis_SI_Process/tests/test_scale_stress_v4.py \
  Dis_SI_Process/tests/test_training_cost_v4.py
```

The composed preview must also be inspected at final 183-mm print width for
label, legend, panel-boundary, and native/throughput-region collisions.
# Figure 5 V4.2

V4.2 is the additive correction that restores panel d to canonical training update time (`ms/update`). Build it with:

```bash
conda run -n fig python Dis_SI_Process/scripts/build_figure5_v42.py --strict-formal
```

The V4 single-stage timing coordinates are preserved exactly. Geo-FNO is admitted only through the clean two-GPU DDP global-batch-192 timing run; Latent FM remains unavailable rather than combining unlike stage times.

### Metric-matched Zero-H Figure 5 V4.2 backup

The replacement Zero-H-balanced backup mirrors formal Figure 5 panels a–d while using only checkpoints and measurements from `1_SubTask_SuperResolution` recipe `4_ZeroH_Balanced`. Panels a/b contain the two stochastic adopted models; panels c/d contain all four adopted models. Build it without fallbacks using:

```bash
conda run -n fig python Dis_SI_Process/scripts/build_figure5_zeroh_matched_v42.py --strict-formal
```

# Figure 5 V4 source schema

This contract is additive to the V3 workflow. V3 files used by panels a/b/c
remain in `results/ValidationV3`; V4 panel-d and panel-e files live under
`results/ValidationV4` and are never resolved from an older root.

## Panel d: training-compute source

Run directory:

```text
Dis_SI_Process/results/ValidationV4/TrainingCost/<run_id>/
  manifest.json
  qa.json
  training_cost_summary.csv
  training_stage_summary.csv
  training_update_repeats.csv
  gpu_state_before.txt
  gpu_state_after.txt
```

`manifest.json` must have `schema_version:
figure5-validation-v4-training-cost-1`, `status: complete`, `formal: true`,
and either `metric_name` or `metric: {name, unit}`. The declared metric must be
one of `total_gpu_hours`, `replay_equivalent_gpu_hours`, or
`training_update_time_ms`. Replay-equivalent GPU-hours additionally require a
passing `promotion_gate` with `validated: true` and a tolerance no larger than
the configured 25%.

The promoted V4 run uses `training_update_time_ms`: the median synchronized
canonical forward/loss/backward/gradient-clip/optimizer update at each stage's
adopted batch/query configuration. Dataset I/O, loader workers, host transfer,
validation, checkpointing, and epoch scheduler work are outside the boundary.
Each successful stage has 20 warmups and 100 measured updates arranged as 10
blocks of 10. The stability gate compares the first-five and last-five block
medians and requires a relative difference no larger than 25%. Because adopted
batch sizes differ, panel d is a descriptive checkpoint footprint rather than
a batch-normalized or matched-budget causal comparison.

The summary table has one row per adopted method and these columns:

```text
method,status,cost_value,cost_low,cost_high,training_update_time_ms,
error,error_ci_low,error_ci_high,checkpoint_path,checkpoint_sha256,
training_cost_basis,unavailable_reason,stage_count,total_update_count
```

Rows with `status=ok` are plotted; unavailable methods must be documented in
the manifest/companion rather than represented by invented coordinates.
Latent FM has two required, unlike training stages, so both are measured in
`training_stage_summary.csv` but no single method-level update-time scalar is
drawn. Geo-FNO's canonical adopted batch of 192 exceeded the 47.38-GiB device
capacity and is likewise recorded as unavailable rather than rerun at a
smaller, noncanonical batch. `training_update_repeats.csv` retains the 100
timings, losses, and peak allocated memory values for each successful stage;
the stage summary records the explicit failure for the attempted Geo-FNO
stage. The error
columns are the frozen Figure 4 unobserved-field relative-L2 coordinate.

## Panel e: high-resolution scalability source

Run directory:

```text
Dis_SI_Process/results/ValidationV4/ScaleStress/<run_id>/
  manifest.json
  qa.json
  scale_stress_summary.csv
  native_query_support_audit.csv
  boundary_summary.csv
  query_coordinates_manifest.csv
```

`manifest.json` must have `schema_version:
figure5-validation-v4-scale-stress-1`, `status: complete`, `formal: true`, and
the frozen V4 protocol under `protocol`. The protocol declares
`predeclared_query_counts=[100000,250000,500000,1000000,2000000,4000000]`, an
adaptive `candidate_query_counts` ending at the global 8M cap, native
`N=40300`, and a clean-GPU warm model-core timing boundary. The manifest also
contains `dummy_query_spec` with
`generator=torch.quasirandom.SobolEngine`,
`sequence_policy=exact_sensor_prefix_then_sobol_suffix`, the exact 256-sensor
prefix, and `dummy_query_spec_sha256`. The QA JSON must pass the common-query,
throughput-only/no-accuracy, geometry-separation, identity, and clean-GPU
checks. A method may terminate at the first declared boundary; it need not
have rows for unattempted counts after that boundary.

`native_query_support_audit.csv` is the architecture audit and has `method`,
`native_query_supported`, `query_scaling_eligible`, `native_only`,
`decision_basis`, and canonical source/evidence fields. A variable-query basis
must describe canonical arbitrary query evaluation; full-grid reconstruction
followed by slicing is rejected. Its eligible method set must agree with the
validated V3 support table.

`scale_stress_summary.csv` is the V4 throughput-only table and has one combined
row per attempted `(method,N)` with at least:

```text
method,N,status,median_latency_ms,latency_q25_ms,latency_q75_ms,
peak_allocated_mib,query_sha256,throughput_only,accuracy_claim
```

Successful rows are `status=ok`; the first OOM/runtime/memory boundary may be
recorded as `boundary_failure`/`failed` and may have no numeric metric. Counts
must form a prefix of the predeclared/adaptive grid and no row may occur after
the first failure. Every V4 row has `N>40300`,
`query_region=throughput_only` in the consumer's in-memory export, and no
accuracy coordinate.

`query_coordinates_manifest.csv` records the shared per-count
`query_sha256/spec_sha256`, Sobol generator, sensor-prefix count, and
throughput-only/no-accuracy flags. `boundary_summary.csv` records one row per
variable-query method with `method`, `largest_success_N`, `first_failure_N`,
and `termination_reason`. The stress grid is attempted in order for every
eligible method; the largest success and first failure are retained even when
the curve terminates at a hardware boundary. The consumer combines these V4
rows in memory with the validated V3 `query_latency_summary.csv` and
`memory_summary.csv` native prefix:

```text
V3: N={1024,4096,16384,40300} for query-evaluable methods;
V3: N=40300 only for fixed-grid methods;
V4: N>40300 stress rows only.
```

Fixed-grid methods therefore receive only their V3 native `N=40300` open
marker; no V4 curve is fabricated for them.

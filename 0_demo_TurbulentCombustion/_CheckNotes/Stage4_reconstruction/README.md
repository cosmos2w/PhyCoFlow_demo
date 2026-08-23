# Stage 4 cached-streamed reconstruction

Checkpoint: Round-1 optimized `best.pt` (GL_rbf_ENH + topk_rbf_glres + KeOps).
GPU 0 had the same pre-existing ~10.6 GiB high-utilization co-tenant noted in
Stages 1 and 3, so wall times are controlled relative diagnostics. CUDA peak
allocations are process-local.

## Scaling result

Batch size 1, 256 observations, Euler, two steps/NFEs, FP32. Cached execution
uses 8,192-query chunks and `static_features`.

| N | mode | wall s | s / million points / NFE | peak allocated MB | query cache MB |
|---:|---|---:|---:|---:|---:|
| 40,300 | legacy_full | 1.169 | 14.50 | 467.0 | 0.0 |
| 40,300 | cached_streamed | 0.114 | 1.41 | 387.0 | 88.5 |
| 250,000 | legacy_full | 4.494 | 8.99 | 2,782.0 | 0.0 |
| 250,000 | cached_streamed | 0.671 | 1.34 | 852.4 | 549.3 |
| 1,000,000 | cached_streamed | 2.675 | 1.34 | 2,958.4 | 2,197.3 |

At 250k, cached streaming is 6.70x faster and uses 69.4% less peak allocation.
The one-million-point run completes without a legacy-full attempt. Its peak is
dominated by the explicit FP32 static cache (2.20 GiB); dynamic hidden execution
is chunked end-to-end and no full `[N, hidden]` dynamic head tensor is retained.
Cache construction preallocates and fills full cache tensors by chunk, avoiding a
second list-plus-concatenate copy.

## Numerical equivalence

The focused deterministic matrix covers:

- `topk_rbf` and `topk_rbf_glres`;
- Euler and Heun;
- 1, 2, and 4 steps;
- `none`, `default_hard`, `endpoint`, and `endpoint_smooth` consistency;
- `none`, `geometry`, and `static_features` cache levels.

All 21 focused tests pass. A real validation snapshot using the checkpoint,
Heun, two steps, and endpoint-smooth consistency gives:

- max absolute cached-minus-legacy difference: `3.09944e-6`;
- mean absolute difference: `2.48932e-7`;
- legacy relative L2: `0.7297682166`;
- cached-streamed relative L2: `0.7297682166`.

The condition encoder is asserted to execute once for a four-step Heun
trajectory, rather than eight times.

## Exact benchmark command

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n phycoflow_env \
  python src/benchmark_pointcloud_reconstruction.py \
  --config _CheckNotes/Round1_runs/optimized/ffm_tc_pointcloud_DemoN9012_20260820_175500/run_config.yaml \
  --checkpoint _CheckNotes/Round1_runs/optimized/ffm_tc_pointcloud_DemoN9012_20260820_175500/best.pt \
  --stats-path Save_TrainedModel/ffm_tc_pointcloud_DemoN51_20260718_083538/dataset_stats.pt \
  --device cuda:0 --n-points 40300 250000 1000000 --legacy-max-points 250000 \
  --n-obs 256 --n-steps 2 --chunk-size 8192 --cache-level static_features \
  --output-csv _CheckNotes/Stage4_reconstruction/reconstruction_scaling.csv \
  --output-json _CheckNotes/Stage4_reconstruction/reconstruction_scaling.json
```

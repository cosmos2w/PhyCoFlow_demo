# Stage 5 large-effective-query training

Checkpoint/model: Round-1 optimized GL_rbf_ENH + topk_rbf_glres + KeOps.
Benchmark: batch size 1, 256 observations, FP32, one warm-up plus three measured
Adam steps on GPU 0. GPU 0 retained the documented ~10.6 GiB high-utilization
co-tenant, so wall times are relative diagnostics; CUDA peaks are process-local.

## Scaling

| Effective queries | query microbatch | step ms | peak allocated MB | queries/s |
|---:|---:|---:|---:|---:|
| 4,096 | monolithic | 58.49 | 257.0 | 70,024 |
| 16,384 | monolithic | 190.79 | 799.5 | 85,873 |
| 16,384 | 4,096 | 234.24 | 320.9 | 69,945 |
| 16,384 | 8,192 | 209.35 | 500.7 | 78,260 |
| 65,536 | monolithic | 702.36 | 3,025.9 | 93,308 |
| 65,536 | 4,096 | 888.44 | 323.7 | 73,765 |
| 65,536 | 8,192 | 783.22 | 513.0 | 83,675 |

At 65,536 effective queries, microbatch 4,096 lowers peak allocation by 89.3%
for a 26.5% wall-time cost. Microbatch 8,192 lowers peak by 83.0% for an 11.5%
time cost. More importantly, the same 4,096 execution chunk changes only from
320.9 MB at 16,384 effective queries to 323.7 MB at 65,536; activation memory is
controlled by the execution chunk rather than total supervision.

Mean losses for matched RNG sequences agree closely:

- 16,384: monolithic `2.087351839`, micro-4k `2.087351918`, micro-8k `2.087352037`;
- 65,536: monolithic `2.069149295`, micro-4k `2.069149295`, micro-8k `2.069149335`.

## Mandatory equivalence test

The deterministic test uses 31 effective queries and microbatch size 7, so the
last chunk has only three points. It checks:

- total RF loss;
- all parameter gradients, including learnable RBF sigma;
- one clipped Adam update for every parameter;
- validation loss without gradients;
- exactly one RFF prior call spanning all 31 coordinates.

Raw gradients agree with `rtol=8e-5, atol=2e-7`. Adam amplifies a few ~1e-9
summation-order differences in near-zero attention-bias gradients; the largest
post-update absolute delta is `5.8e-6`. The test uses this measured FP32 bound.

## Exact command

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n phycoflow_env \
  python research/benchmarks/benchmark_pointcloud_query_microbatch.py \
  --config _CheckNotes/Round1_runs/optimized/ffm_tc_pointcloud_DemoN9012_20260820_175500/run_config.yaml \
  --checkpoint _CheckNotes/Round1_runs/optimized/ffm_tc_pointcloud_DemoN9012_20260820_175500/best.pt \
  --device cuda:0 --n-query 4096 16384 65536 --microbatch 4096 8192 \
  --n-obs 256 --batch-size 1 --iterations 3 --warmup 1 \
  --output-csv _CheckNotes/Stage5_query_microbatch/query_microbatch_scaling.csv \
  --output-json _CheckNotes/Stage5_query_microbatch/query_microbatch_scaling.json
```

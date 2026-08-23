# Stage 3 PointCloud FFM scaling report

The real dataset contains 40,300 points. Rows at 250k and 1M use a tiled real
snapshot in host memory; their `read_ms` is an in-memory clone and is not labeled
as HDF5 throughput. Model rows use the active GL_rbf_ENH + topk_rbf_glres +
KeOps configuration with synthetic 3-D tensors and batch size 1.

GPU 0 had a pre-existing ~10.6 GiB, high-utilization process. The goal permits
co-location when memory fits, so wall times are diagnostic rather than clean
exclusive-GPU absolutes. CUDA peak numbers are process-local.

## Data-path scaling (mean over M=256/512/1024)

| N_full | N_query | pre-model ms | read ms | normalize ms | host RSS MB | selected GPU input MB |
|---:|---:|---:|---:|---:|---:|---:|
| 40,300 | 4,096 | 17.99 | 1.26 | 2.82 | 642.0 | 0.582 |
| 40,300 | 16,384 | 23.56 | 2.16 | 4.85 | 646.0 | 2.082 |
| 40,300 | 40,300 | 21.37 | 1.81 | 6.96 | 652.2 | 5.001 |
| 250,000 | 4,096 | 18.74 | 2.39 | 2.21 | 657.5 | 0.582 |
| 250,000 | 16,384 | 29.92 | 2.93 | 7.19 | 657.5 | 2.082 |
| 250,000 | 65,536 | 47.43 | 1.37 | 8.73 | 667.6 | 8.082 |
| 1,000,000 | 4,096 | 24.44 | 7.64 | 2.87 | 694.3 | 0.582 |
| 1,000,000 | 16,384 | 33.55 | 5.19 | 7.35 | 684.0 | 2.082 |
| 1,000,000 | 65,536 | 45.73 | 4.82 | 10.72 | 692.1 | 8.082 |

## Model execution scaling

| N_query | M | forward ms | backward ms | step ms | peak allocated MB | queries/s |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 256 | 43.55 | 14.97 | 61.15 | 255.4 | 66,988 |
| 4,096 | 512 | 44.79 | 14.91 | 62.29 | 260.4 | 65,756 |
| 4,096 | 1,024 | 47.43 | 14.50 | 64.53 | 271.1 | 63,479 |
| 16,384 | 256 | 152.29 | 39.48 | 194.35 | 797.9 | 84,300 |
| 16,384 | 512 | 150.26 | 35.46 | 188.32 | 802.9 | 87,000 |
| 16,384 | 1,024 | 179.53 | 36.01 | 218.56 | 812.1 | 74,965 |
| 65,536 | 256 | 586.05 | 124.42 | 713.05 | 3022.7 | 91,909 |
| 65,536 | 512 | 593.61 | 113.21 | 709.39 | 3027.6 | 92,384 |
| 65,536 | 1,024 | 684.11 | 114.56 | 801.20 | 3036.7 | 81,798 |

## Interpretation

- At fixed 4,096 queries, expanding the full mesh from 40.3k to 1M changes
  mean pre-model time from 17.99 ms to 24.44 ms and host RSS from 642.0 MB
  to 694.3 MB. Selected GPU inputs remain 0.582 MB.
- The model step is already 61.15–64.53 ms at 4,096 queries, and reaches
  709.39–801.20 ms at 65,536 queries. At 1M/65,536, the data path averages
  45.73 ms while the model requires 0.71–0.80 s: query-model work dominates.
- Increasing observations from 256 to 1,024 raises model step time by 5.5%
  at 4,096 queries and 12.4% at 65,536 queries. KeOps prevents pairwise-memory
  explosion, but exact top-k search still adds work.
- Model peak allocation rises from 255–271 MB at 4,096 queries to about
  3.02–3.04 GB at 65,536 queries. This near-linear activation scaling motivates
  Stage 4 end-to-end inference streaming and Stage 5 training microbatching.

## Exact commands

See the command blocks below; both used the project `phycoflow_env` and GPU 0.

```bash
python src/benchmark_pointcloud_scaling.py --config Save_config/config_pointcloud_ffm.yaml \
  --stats-path Save_TrainedModel/ffm_tc_pointcloud_DemoN51_20260718_083538/dataset_stats.pt \
  --classes data --n-full 40300 250000 1000000 --n-query 4096 16384 65536 \
  --n-obs 256 512 1024 --data-batch-size 4 --iterations 3 --warmup 1 \
  --device cuda:0 --output-csv _CheckNotes/Stage3_scaling/data_scaling.csv \
  --output-json _CheckNotes/Stage3_scaling/data_scaling.json

python src/benchmark_pointcloud_scaling.py --config Save_config/config_pointcloud_ffm.yaml \
  --stats-path Save_TrainedModel/ffm_tc_pointcloud_DemoN51_20260718_083538/dataset_stats.pt \
  --classes model --n-query 4096 16384 65536 --n-obs 256 512 1024 \
  --model-batch-size 1 --iterations 3 --warmup 1 --device cuda:0 \
  --output-csv _CheckNotes/Stage3_scaling/model_scaling.csv \
  --output-json _CheckNotes/Stage3_scaling/model_scaling.json
```

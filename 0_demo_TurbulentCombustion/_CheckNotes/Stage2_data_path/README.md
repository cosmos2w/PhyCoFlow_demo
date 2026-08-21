# Stage 2 data-path benchmark

Dataset: `Dataset/Merged_COTU0U1P.h5` (`N_full=40,300`, contiguous, uncompressed).
Hardware: physical GPU 0. Each normalization mode ran in a separate process.

Common settings: batch 4, 256 observations per sample, two warm-up iterations,
eight measured iterations, no DataLoader workers. The two CSV files contain the
phase timings and GPU-memory observations.

## Results

| N_query | normalization | normalize ms | materialize ms | pre-model ms | total ms |
|---:|---|---:|---:|---:|---:|
| 4,096 | full after read | 2.600 | 0.536 | 9.559 | 10.544 |
| 4,096 | selected after full read | 1.866 | 0.642 | 7.688 | 8.025 |
| 16,384 | full after read | 1.773 | 0.890 | 11.226 | 11.776 |
| 16,384 | selected after full read | 3.930 | 0.873 | 12.592 | 13.095 |

Maximum resident set size from `/usr/bin/time -v`:

- full normalization: 610,900 KiB;
- selected normalization: 605,232 KiB.

The selected path is the correct default for the active 4,096-query workload and
for meshes where the selected union is sparse relative to `N_full`. At 16,384
queries on this 40,300-point dataset, it crosses over and becomes slower. Stage 3
therefore sweeps `N_full` and `N_query` independently.

## Exact commands

```bash
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v conda run --no-capture-output -n phycoflow_env \
  python src/benchmark_pointcloud_data_path.py \
  --config Save_config/config_pointcloud_ffm.yaml \
  --stats-path Save_TrainedModel/ffm_tc_pointcloud_DemoN51_20260718_083538/dataset_stats.pt \
  --output _CheckNotes/Stage2_data_path/optimized_fullnorm.csv \
  --profiles optimized_fullnorm --batch-size 4 --n-query-points 4096 16384 \
  --n-obs 256 --iterations 8 --warmup 2 --num-workers 0 --device cuda:0

CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v conda run --no-capture-output -n phycoflow_env \
  python src/benchmark_pointcloud_data_path.py \
  --config Save_config/config_pointcloud_ffm.yaml \
  --stats-path Save_TrainedModel/ffm_tc_pointcloud_DemoN51_20260718_083538/dataset_stats.pt \
  --output _CheckNotes/Stage2_data_path/optimized_selectednorm.csv \
  --profiles optimized --batch-size 4 --n-query-points 4096 16384 \
  --n-obs 256 --iterations 8 --warmup 2 --num-workers 0 --device cuda:0
```

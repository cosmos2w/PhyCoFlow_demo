# Archived PointCloudFFM benchmarks

These programs preserve the Stage 2–7 performance and equivalence protocols.
They are not required for ordinary training, reconstruction, or evaluation.
Each remains runnable from the demo root; for example:

```bash
python research/benchmarks/benchmark_pointcloud_scaling.py --help
python research/benchmarks/benchmark_pointcloud_stage7.py --help
```

The local `_bootstrap.py` adds the runtime `src/` directory without depending
on the caller's working directory. Historical evidence remains under
`_CheckNotes/`, and `research/MAP.csv` records the old-to-new organization.

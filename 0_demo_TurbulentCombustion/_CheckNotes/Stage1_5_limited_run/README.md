# Stage 1–5 limited validation package

This package validates execution and scaling only. It is not a formal accuracy
campaign or a hyperparameter study.

- `control.yaml`: 8 epochs, effective 4,096 queries, monolithic query execution.
- `large_effective_query.yaml`: 5 epochs, effective 16,384 queries, 4,096-query
  execution chunks.
- Both use the same GL_rbf_ENH + topk_rbf_glres model, dataset, Adam settings,
  batch size 96, optimized selected-normalization data path, and cached-streamed
  final reconstruction at NFE 1 and 2.
- Validation runs at epoch 1 and final. Final checkpoints are additionally
  evaluated on the Stage-1 fixed manifest with controlled RF RNG.
- `analyze.py` combines loss, epoch-time, sampled peak-memory, fixed-manifest,
  reconstruction, and the Stage-4 one-million-query stress result.

Run from the repository root:

```bash
bash _CheckNotes/Stage1_5_limited_run/launch.sh
```

GPU 0 is used directly. The goal explicitly permits co-location if memory fits;
record the initial co-tenant state when interpreting wall time.

## Completed results

GPU 0 started with a pre-existing 10,636 MiB/100%-utilization process.

| Run | epochs | effective / micro queries | final train | final val | steady epoch s | sampled peak MB | fixed-manifest mean |
|---|---:|---|---:|---:|---:|---:|---:|
| Control A | 8 | 4,096 / monolithic | 1.044623 | 1.038733 | 58.72 | 20,745.5 | 0.948706 |
| Large-query B | 5 | 16,384 / 4,096 | 1.188536 | 1.197108 | 108.69 | 26,267.3 | 1.053499 |

The first four epoch losses closely track:

| epoch | Control A | Large-query B |
|---:|---:|---:|
| 1 | 1.744952 | 1.744743 |
| 2 | 1.388033 | 1.389656 |
| 3 | 1.301028 | 1.297611 |
| 4 | 1.226664 | 1.227312 |

B is 3.9% higher at epoch 5 (`1.188536` vs Control `1.143578`) but remains
stable and shows no obvious convergence failure. Fixed-manifest losses are not a
same-epoch quality comparison because Control is epoch 8 and B is epoch 5.

Both runs completed cached Euler NFE-1/NFE-2 reconstruction with 256 hard
sensors and zero sensor-consistency error. The independent one-million-query
stress completed at 2,958.4 MB process-local peak and 2.675 s wall time.

### Recovery note

Control completed epoch-8 training, validation, and checkpoint saving, then its
first final visualization exposed that `visualize_reconstruction()` still used
eager dataset indexing while the optimized dataset defers field reads. The
helper now calls `get_full_snapshot()` when available. Control reconstruction
was recovered from the saved epoch-8 checkpoint without retraining. Because the
history logger ran after visualization, `loss_history.csv` contains seven rows;
the epoch-8 train/validation values are preserved in the log and checkpoint and
are parsed by `analyze.py`. B then completed normally through all five history
rows using the fixed helper.

Generated run directories and terminal logs total roughly 155 MB and are kept
local via this package's `.gitignore`. Compact configs, controlled evaluation
rows, initial GPU state, scripts, and `summary.json` are committed.

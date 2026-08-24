# Stage 8 real-data smoke

The frozen and optimized runs used the same seed-42 GL_rbf_CQ configuration,
B128/Q4096, a 2048-query training microbatch, RFF objective, optimizer, cosine
horizon 1000, EMA, data path, and 192–384 temperature sensors. Only condition
attention execution differed. Reconstruction was disabled for this smoke.

| mode | epochs compared | mean train seconds/epoch | epoch-3 train loss | epoch-3 EMA validation loss |
|---|---:|---:|---:|---:|
| legacy MHA + full padding | 3 | 26.826 | 1.427770 | 1.933348 |
| cached K/V + full padding | 3 | 25.892 | 1.436426 | 1.933353 |

Cached/full was 3.48% faster over the first three real training epochs. The
validation difference at epoch 3 was `4.65e-6` absolute. Both runs were finite
and stable. The cached run was then resumed from `last.pt` for epoch 4:

- the log explicitly reported resume from epoch 3 and start at epoch 4;
- optimizer state was populated;
- scheduler `last_epoch` became 4;
- EMA remained enabled at decay 0.999 and reached 284 updates;
- epoch-4 train/EMA-validation losses were 1.316036 / 1.899658.

Run directories:

- `legacy_full_DemoN9881_20260824_141950/`
- `cached_full_DemoN9882_20260824_142126/`

# Figure 5d — Accuracy and computational footprint

This accuracy-first D1 graphical scorecard keeps every quantity in a separate aligned column and uses no bubble area, weighted score, rank average, or stage-count column.

| Method | Error | Train time (ms/update) | Train memory (GiB) | Inference time (ms) | Model state / inference peak (MiB) |
|---|---:|---:|---:|---:|---:|
| DMF-Gen | 0.117 | 112.2 | 7.9 | 16.69 | 24.8 / 417.6 |
| Senseiver | 0.143 | 49.9 | 3.8 | 8.30 | 31.8 / 536.7 |
| SiT | 0.210 | 661.8 | 14.5 | 20.99 | 39.9 / 82.4 |
| Geo-FNO | 0.230 | 235.7 | 9.3 | 3.41 | 19.7 / 122.5 |
| FFM-Perceiver | 0.348 | 109.2 | 4.7 | 23.09 | 20.1 / 312.5 |
| FFM-FNO | 0.390 | 249.8 | 9.2 | 8.70 | 19.7 / 149.7 |
| MLP-RBF | 0.396 | 24.9 | 1.9 | 3.14 | 2.3 / 280.7 |
| Latent FM | 0.453 | 90.7 | 4.1 | 10.17 | 337.8 / 392.1 |

## Protocol and source notes

- Accuracy and native warm inference latency: `Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_plot_source_common_b32.csv`; error is the frozen 1,000-state mean unobserved-field relative L2 with temporal-bootstrap 95% interval, while latency is median with IQR.
- Training time/memory: `Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_stage_source_common_b32.csv`; common B=32, M=256, float32, synchronized update timing, one clean GPU. Query-evaluable models use 4,096 training targets and native-grid architectures use 40,300, so values are descriptive method-native workloads rather than an asymptotic or matched-budget comparison.
- Inference memory: `Dis_SI_Process/results/ValidationV51/InferenceMemory/inference_memory_native_v51_20260903_094021/inference_memory_summary.csv`; B=1, M=256, N=40,300, float32, `torch.inference_mode`, 5 warmups and 10 repeats. Open markers show unique parameters plus persistent buffers; filled markers show process-local peak allocated memory during inference. The benchmark allowed unrelated shared-GPU work and therefore makes no timing claim.
- Latent FM has two required non-concurrent stages. For the one-row main panel, training time shows the larger stage-2 median (90.7 ms/update) and training memory shows the larger stage-1 peak (4.1 GiB). They are per-column maxima, not simultaneous or additive costs.

The scorecard is chosen because reconstruction accuracy remains the first and widest quantitative column, while offline and online costs stay distinct, inspectable, and correctly qualified.

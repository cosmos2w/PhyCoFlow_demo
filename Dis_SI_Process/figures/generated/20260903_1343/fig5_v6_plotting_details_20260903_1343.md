# Figure 5 V6 plotting and quantitative reference

This document records the plotting design, displayed quantities, model identities, evidence sources, and interpretation limits for `fig5_composed_v6_20260903_1343.svg`. It is intended to support manuscript caption, Results, Methods, and Supplementary Information writing. Release `20260903_1343` is a formatting-only refinement: it reuses accepted V5/V5.1 evidence and does not rerun model inference, bootstrapping, training, or validation.

## Figure-level scientific logic

Figure 5 supports the following evidence chain:

1. **Panel a:** the conditional predictive distribution is assessed with normalized CRPS.
2. **Panel b:** ensemble spread is tested as an indicator of statewise reconstruction difficulty.
3. **Panel c:** the association in panel b is translated into uncertainty-guided selective reconstruction.
4. **Panel d:** reconstruction accuracy is placed first, followed by separately measured training and inference costs in physical units.

The common task is turbulent-combustion missing-channel reconstruction under `Cond_T`. Each method receives `M = 256` temperature observations and reconstructs the four unobserved fields `Y_CH4`, `Y_CO`, `U1`, and `p` on the native `N = 40,300`-point grid. Fieldwise metrics are macro-averaged with equal weight, 0.25 per unobserved field.

Panels a–c use the five methods with empirical conditional ensembles: DMF-Gen, FFM-FNO, FFM-Perceiver, Latent FM, and SiT. Panel d includes all eight evaluated methods by adding Senseiver, Geo-FNO, and MLP-RBF. This difference in method coverage is intentional.

## Model identities and visual encoding

Checkpoint identity was verified by SHA-256 in the accepted benchmark manifests. Hash prefixes below are sufficient for cross-reference; the source manifests contain the complete hashes and checkpoint paths.

| Display name | Model family | Backbone | Evaluation weights | Panels | Marker | Color | SHA-256 prefix |
|---|---|---|---|---|---|---|---|
| DMF-Gen | `pointcloud_ffm` | `GL_rbf_ENH` | checkpoint model weights | a–d | circle | red `#E63946` | `857a505ff96c` |
| FFM-FNO | `pointcloud_ffm` | `fno` | checkpoint model weights | a–d | square | navy `#1D3557` | `9819ad702a94` |
| FFM-Perceiver | `pointcloud_ffm` | `perceiver` | checkpoint model weights | a–d | diamond | blue `#457B9D` | `c35041c26835` |
| Latent FM | `latent_fm` | `latent_fm` | materialized EMA | a–d | upward triangle | purple `#6A4C93` | `d2257080b437` |
| SiT | `sit` | `sit` | materialized EMA | a–d | downward triangle | light purple `#A28BC4` | `1d20713b43bc` |
| Senseiver | `senseiver` | `senseiver` | checkpoint model weights | d | hexagon | grey-blue `#8D99AE` | `b055ccc8ad86` |
| Geo-FNO | `geofno` | `geofno` | checkpoint model weights | d | filled-X shape | orange `#F4A261` | `56208012cebc` |
| MLP-RBF | `mlp_rbf` | `mlp_rbf` | checkpoint model weights | d | filled-plus shape | teal `#2A9D8F` | `a528358e6db1` |

Outside the inference-memory subplot, every method marker retains its method-specific shape but is drawn with a white fill and a 1.10-pt method-color outline. Marker size is 5.0 pt. In inference memory, the same method-specific shape is used at both endpoints: the **filled** symbol is model parameters plus persistent buffers (`Model`), and the **hollow** symbol is peak allocated memory during inference (`Peak`). A thin line joins the endpoints. `Model` and `Peak` are identified once using the DMF-Gen row and thin leader lines.

## Panel a — normalized CRPS

### Scientific quantity

The x-axis is **Normalized CRPS**; lower values indicate better predictive-distribution quality. For each unobserved field, empirical CRPS is divided by the frozen training-set standard deviation, averaged spatially, and then macro-averaged equally across the four fields.

The UQ cohort contains 200 paired held-out temporal states and 64 shared-seed ensemble draws per state. The formal summary is the mean statewise normalized CRPS with a 95% temporal moving-block-bootstrap interval using 2,000 replicates and block length 25.

### Graphical design

- Rows, from top to bottom: DMF-Gen, FFM-FNO, FFM-Perceiver, Latent FM, SiT.
- Every statewise value is shown as a small, deterministically jittered point.
- The translucent box spans the first to third quartile; its internal line is the median; conventional 1.5-IQR whiskers summarize the statewise distribution. Outliers are not redrawn by the boxplot because the underlying points are already shown.
- The large hollow method symbol is the formal mean. The capped horizontal line through it is the accepted 95% bootstrap interval.
- The DMF-Gen row has a pale red background highlight.
- Linear x range: 0.00–0.86; labeled ticks: 0.0, 0.2, 0.4, 0.6, 0.8.

### Quantitative values

| Method | Mean normalized CRPS | 95% CI |
|---|---:|---:|
| DMF-Gen | 0.0667 | [0.0640, 0.0694] |
| FFM-FNO | 0.3989 | [0.3739, 0.4307] |
| FFM-Perceiver | 0.2596 | [0.2476, 0.2723] |
| Latent FM | 0.3711 | [0.3544, 0.3896] |
| SiT | 0.0999 | [0.0970, 0.1030] |

DMF-Gen has the lowest accepted normalized CRPS, followed by SiT. CRPS evaluates the predictive distribution but does not by itself demonstrate calibration; the separate accepted reliability analysis finds finite-ensemble underdispersion and belongs in SI.

## Panel b — spread–error association

### Scientific quantity

The x-axis is **Spearman ρ** between macro normalized ensemble spread and macro ensemble-mean relative-L2 reconstruction error across the same 200 states. A positive value means that higher empirical spread tends to occur on states with higher reconstruction error. This is an uncertainty-informativeness result, not proof of calibration, causal error prediction, or a Bayesian posterior interpretation.

### Graphical design

- The method rows are identical to panel a and geometrically aligned with it; redundant method labels are suppressed in panel b.
- The cloud represents the accepted 2,000 temporal moving-block-bootstrap Spearman estimates. For compact display, 320 evenly indexed bootstrap values per method are shown as jittered points, while the distribution and interval calculations use all 2,000 values.
- The translucent box spans the bootstrap interquartile range. Its whiskers are the 2.5th and 97.5th percentiles.
- The large hollow symbol is the full-sample Spearman estimate. The capped horizontal line is its accepted 95% moving-block-bootstrap interval.
- The vertical dashed reference line marks ρ = 0.
- Linear x range: −0.28–0.80; labeled ticks: −0.2, 0.0, 0.2, 0.4, 0.6, 0.8.

### Quantitative values

| Method | Full-sample Spearman ρ | 95% CI |
|---|---:|---:|
| DMF-Gen | 0.654 | [0.560, 0.721] |
| FFM-FNO | 0.183 | [−0.004, 0.359] |
| FFM-Perceiver | 0.215 | [0.080, 0.348] |
| Latent FM | −0.033 | [−0.164, 0.106] |
| SiT | 0.261 | [0.103, 0.384] |

DMF-Gen has the largest positive association. The accepted intervals for DMF-Gen, FFM-Perceiver, and SiT lie above zero; the intervals for FFM-FNO and Latent FM include zero.

## Panel c — normalized selective-reconstruction risk

### Scientific quantity

States are ranked within each method by ascending macro normalized ensemble spread. At retained fraction `r`, the least-uncertain `ceil(200r)` states are retained and their mean macro ensemble-mean relative-L2 error is computed. The plotted quantity is

`relative retained-set error = R_m(r) / R_m(1)`,

where `R_m(1)` is the same method's full 200-state cohort error. Lower is better. Normalization forces every method to end at 1.0 and deliberately removes absolute-accuracy differences already represented in panels a and d.

The coverage grid is 0.20, 0.30, ..., 1.00, corresponding to 40, 60, ..., 200 retained states. Exact evaluated points are connected without fitted smoothing. Bands are 95% temporal moving-block-bootstrap intervals from 2,000 replicates with block length 25; state ranking is recomputed inside every bootstrap replicate.

### Graphical design

- Five method-color lines with method-specific hollow markers and translucent confidence bands.
- A horizontal dashed reference marks relative error 1.0.
- Linear x range: 0.18–1.02; labeled ticks: 0.2, 0.4, 0.6, 0.8, 1.0.
- Linear y range: 0.835–1.025; labeled ticks: 0.85, 0.90, 0.95, 1.00.
- The retained direct annotation `DMF AURC = 0.741` reports the trapezoidal area over the evaluated 0.20–1.00 grid. AURC is not divided by the grid width of 0.8.
- The composed release has no separate legend; method identities are carried consistently by color and shape from panels a and d.

### Exact central curve values

| Method | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DMF-Gen | 0.878 | 0.879 | 0.896 | 0.907 | 0.922 | 0.943 | 0.954 | 0.970 | 1.000 | 0.741 |
| FFM-FNO | 0.990 | 0.991 | 0.993 | 0.992 | 0.991 | 0.993 | 0.995 | 0.997 | 1.000 | 0.795 |
| FFM-Perceiver | 0.967 | 0.975 | 0.985 | 0.988 | 0.991 | 0.995 | 0.999 | 0.999 | 1.000 | 0.792 |
| Latent FM | 0.998 | 0.997 | 1.003 | 1.004 | 1.003 | 1.002 | 1.004 | 1.003 | 1.000 | 0.801 |
| SiT | 0.948 | 0.956 | 0.965 | 0.974 | 0.980 | 0.976 | 0.991 | 0.994 | 1.000 | 0.781 |

### Key confidence intervals and 80% selection effect

| Method | 20% risk [95% CI] | 80% risk [95% CI] | Error reduction from 100% to 80% |
|---|---:|---:|---:|
| DMF-Gen | 0.878 [0.845, 0.902] | 0.954 [0.945, 0.968] | 4.6% |
| FFM-FNO | 0.990 [0.965, 1.009] | 0.995 [0.990, 0.998] | 0.5% |
| FFM-Perceiver | 0.967 [0.954, 0.990] | 0.999 [0.994, 1.003] | 0.1% |
| Latent FM | 0.998 [0.983, 1.019] | 1.004 [0.998, 1.009] | −0.4% (slight increase) |
| SiT | 0.948 [0.915, 0.983] | 0.991 [0.980, 0.997] | 0.9% |

The full 200-state errors used only as the normalization denominators are DMF-Gen 0.10994, FFM-FNO 0.36813, FFM-Perceiver 0.33710, Latent FM 0.44130, and SiT 0.19670. These are **not** the panel-d errors, which use a different frozen 1,000-state cohort.

C1b is used in the main figure because it asks the distinct operational question of how well each method's own uncertainty ranks states for selective retention. The absolute C1a version remains appropriate for SI or backup use.

## Panel d — accuracy and computational footprint scorecard

### Overall design

The eight rows are ordered by increasing reconstruction error: DMF-Gen, Senseiver, SiT, Geo-FNO, FFM-Perceiver, FFM-FNO, MLP-RBF, Latent FM. This order makes accuracy visually primary and is not a weighted multi-metric rank. The common y-axis appears only at the far left. Pale horizontal guides connect each method across all five columns, and the DMF-Gen row has a pale red highlight.

The five columns are:

1. **Reconstruction error:** linear relative-L2 scale, 0.08–0.49; ticks 0.1, 0.2, 0.3, 0.4.
2. **Training time:** logarithmic ms/update scale, 20–820; ticks 25, 100, 400.
3. **Training memory:** logarithmic GiB scale, 1.5–18.5; ticks 2, 4, 8, 16.
4. **Inference time:** logarithmic ms scale, 2.5–30; ticks 3, 10, 30.
5. **Inference memory:** logarithmic MiB scale, 1.6–700; ticks 2, 10, 100, 500.

No stage-count column, bubble area, composite score, or lifecycle trajectory is used.

### Exact scorecard quantities

| Method | Relative L2 [95% CI] | Training ms/update [IQR] | Training GiB | Inference ms [IQR] | Model / peak MiB |
|---|---:|---:|---:|---:|---:|
| DMF-Gen | 0.117 [0.115, 0.119] | 112.2 [111.3, 113.0] | 7.9 | 16.7 [16.5, 16.9] | 24.8 / 417.6 |
| Senseiver | 0.143 [0.141, 0.145] | 49.9 [49.4, 50.5] | 3.8 | 8.3 [8.2, 8.6] | 31.8 / 536.7 |
| SiT | 0.210 [0.208, 0.213] | 661.8 [657.1, 666.6] | 14.5 | 21.0 [20.8, 21.4] | 39.9 / 82.4 |
| Geo-FNO | 0.230 [0.227, 0.233] | 235.7 [235.4, 236.3] | 9.3 | 3.4 [3.3, 3.5] | 19.7 / 122.5 |
| FFM-Perceiver | 0.348 [0.345, 0.351] | 109.2 [107.8, 110.2] | 4.7 | 23.1 [22.9, 23.4] | 20.1 / 312.5 |
| FFM-FNO | 0.390 [0.387, 0.392] | 249.8 [249.4, 250.4] | 9.2 | 8.7 [8.5, 9.3] | 19.7 / 149.7 |
| MLP-RBF | 0.396 [0.393, 0.399] | 24.9 [24.6, 25.3] | 1.9 | 3.1 [3.1, 3.2] | 2.3 / 280.7 |
| Latent FM | 0.453 [0.449, 0.457] | 90.7 [89.4, 92.1] | 4.1 | 10.2 [9.9, 10.4] | 337.8 / 392.1 |

### Column-specific statistical and graphical definitions

- **Reconstruction error:** frozen 1,000-state mean unobserved-field relative L2. The capped horizontal line is the temporal-bootstrap 95% interval. The plotted number is printed with three decimals immediately to the right of the interval.
- **Training time:** median synchronized update time, with a horizontal IQR. The plotted number is printed above the marker with zero decimals in the figure.
- **Training memory:** peak allocated CUDA memory for the selected method stage, converted from MiB to GiB by division by 1,024. No interval is drawn. The number is printed above the marker with one decimal.
- **Inference time:** median warm native-resolution latency, with a horizontal IQR. The number is printed above the marker with one decimal.
- **Inference memory:** the line joins model state and process-local peak allocated memory. In this release, the filled endpoint means model parameters plus persistent buffers and the hollow endpoint means peak allocated memory. Numerical values are intentionally omitted from the plot and retained in the table above.

### Training benchmark boundary

- Hardware: one clean NVIDIA RTX 6000 Ada Generation GPU; CUDA 12.1; PyTorch 2.5.1; float32.
- Common batch size: 32; sensor count: 256.
- Warmup: 20 updates. Measurement: 10 blocks × 10 updates = 100 updates.
- Timing includes device-side condition preparation, forward pass, canonical training loss, backward pass, gradient clipping, optimizer step, and native EMA update when applicable.
- Timing excludes dataset I/O, data-loader work, host-to-device transfer, validation, logging, plotting, and checkpointing.
- Query-evaluable methods—DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver—use 4,096 training targets. Native-grid methods—FFM-FNO, Latent FM, SiT, and Geo-FNO—use 40,300 targets. The resource columns therefore describe method-native workloads at a common batch size, not identical asymptotic workloads.
- Latent FM has two required, non-concurrent stages. The time column uses the slower stage-2 median, 90.7 ms/update; the memory column uses the larger stage-1 peak, 4.1 GiB. These per-column maxima are neither simultaneous nor additive.

### Inference-time and inference-memory boundaries

Warm inference latency uses batch size 1, 256 sensors, 40,300 output points, float32, and the `warm_model_core_geometry_persisted` boundary.

The dedicated inference-memory benchmark uses batch size 1, 256 sensors, 40,300 output points, float32, `torch.inference_mode`, five warmups, and ten measured repeats. Peak memory is `torch.cuda.max_memory_allocated` for the process-local inference call. Model state is unique parameter plus persistent-buffer storage used by inference modules. All methods use query chunk size 8,192 in the benchmark configuration. The GPU allowed unrelated shared work, so this dedicated run supports a memory claim only and makes no timing claim.

Inference execution modes were DMF-Gen `cached_streamed` with persistent static-feature caching; FFM-FNO and FFM-Perceiver `legacy_full`; Latent FM, SiT, MLP-RBF, and Geo-FNO native baseline adapters; and Senseiver an unstreamed native baseline adapter.

## Final layout and typography

- Canvas: 183 mm × 128 mm, white background, editable SVG text.
- Composition: panels a, b, and c on the upper row; the five-column panel d spans the lower row.
- Font: Microsoft Arial, explicitly registered from the installed user font directory.
- Panel tags: 8.4 pt bold Arial. Tags a and d share normalized x position 0.025.
- Axis labels: 6.6 pt. Tick labels: 5.8 pt. Panel-d column headings: 6.2 pt semibold with identical title padding, including `Inference memory`.
- Numeric annotations: 4.8 pt for reconstruction error and 4.6 pt for the time/memory scorecard values. `Model` and `Peak` labels are 5.2 pt.
- Top-row axes positions in normalized figure coordinates: a `[0.120, 0.595, 0.250, 0.310]`, b `[0.415, 0.595, 0.270, 0.310]`, c `[0.735, 0.595, 0.260, 0.310]`.
- Panel-d axes span normalized x = 0.120–0.995 and y = 0.070–0.475. Column width ratios are 2.45 : 1.05 : 1.10 : 1.05 : 1.40 with a normalized inter-column gap of 0.025.
- The upper method axis and lower scorecard method axis have the same normalized left boundary, x = 0.120. The upper/lower axes gap is 0.120.
- Descriptive subtitles above panels a–c, the overall panel-d title, and the method legend are intentionally absent.

## Suggested manuscript wording

**Results-level statement.** DMF-Gen achieved the lowest mean normalized CRPS (0.0667, 95% CI 0.0640–0.0694) and the strongest association between ensemble spread and statewise reconstruction error (Spearman ρ = 0.654, 95% CI 0.560–0.721). Retaining the 80% least-uncertain states reduced its cohort-normalized error by 4.6%, compared with reductions of 0.1–0.9% for the other positively selective baselines; Latent FM showed a slight 0.4% increase. On the separate frozen 1,000-state accuracy cohort, DMF-Gen had the lowest mean relative-L2 error (0.117, 95% CI 0.115–0.119), while the scorecard separately exposes its training and inference resource requirements.

**Interpretation guardrails.** Do not describe panel b as calibrated uncertainty or causal failure prediction. Do not compare panel-c full-cohort errors numerically with panel-d errors without stating the different cohort sizes. Do not add the two Latent FM training-stage costs. Do not interpret inference-memory benchmark data as a latency comparison. Do not combine panel-d columns into an undocumented aggregate score.

## Source provenance

| Role | Reused source | Data rows | SHA-256 |
|---|---|---:|---|
| Panels a/b plotted samples and summaries | `Dis_SI_Process/results/derived/20260831_1409/figure5_v5_source.csv` | 11,074 total; 1,005 used by a and 10,005 used by b | `8753eec426d6fbfc83ff5f3a469d05b8b2596dab114f9c28aaa192b973ce1615` |
| Panel c absolute and normalized selective-risk family | `Dis_SI_Process/results/ValidationV51/PanelC/panel_c_exploration_20260902_1129/selective_risk.csv` | 90 total; 45 normalized rows plotted | `6144766d3b9c89027971b310c8a1ba982c1af06ac534826bac4616cacb3b3679` |
| Panel d accuracy and warm inference time | `Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_plot_source_common_b32.csv` | 8 | `3c966c0f75a0b5c927267aceb77bb3625e006af086aaaac69b96df52984cd6e8` |
| Panel d training stages | `Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_stage_source_common_b32.csv` | 9 | `79d3491f9268ba668a2a8a68535aade89e2df7e73f4e6d716cfae71f34407ac4` |
| Panel d inference-memory endpoints | `Dis_SI_Process/results/ValidationV51/InferenceMemory/inference_memory_native_v51_20260903_094021/inference_memory_summary.csv` | 8 | `60af88d21ca33e3585ba9e8b142de1f42f267169a64a56b018c5a2b84529c6b2` |

The accepted QA files for each evidence package report passing status. Release-level QA additionally confirms that every SVG is parseable with editable Arial text, all required labels are present, removed subtitles/legend text are absent, visual QA passed, and the scientific-geometry digest is identical before and after formatting for panels a–d and the composed figure.

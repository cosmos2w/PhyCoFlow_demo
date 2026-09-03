# Figure 5 V6 contract

## Scientific argument

- **Core conclusion:** DMF-Gen produces the most accurate conditional ensembles and the lowest sparse-reconstruction error, while its empirical uncertainty is useful for identifying difficult states and its measured training/inference footprint remains moderate rather than uniformly minimal.
- **Archetype:** quantitative grid with three equal-weight validation panels above one full-width, accuracy-first graphical scorecard.
- **Target/output:** Nature-family main-text figure; editable SVG; 183 mm × 128 mm composed canvas; Python/Matplotlib exclusively in the `fig` environment.

## Panel map and evidence chain

- **a — Probabilistic reconstruction:** accepted V5 statewise normalized CRPS; 200 paired held-out states per generator, formal mean, and temporal moving-block-bootstrap 95% interval.
- **b — Uncertainty tracks difficult states:** accepted V5 spread–error Spearman evidence; 2,000-replicate bootstrap cloud, full-sample estimate and 95% interval, with a zero reference.
- **c — Uncertainty supports selective reconstruction:** accepted V5.1 C1b normalized selective-risk curves; the least-uncertain 20–100% of states are retained and retained-set error is divided by each method's full-cohort error. All curves end at 1.0; lower is better; no logarithmic scale is used.
- **d — Accuracy and computational footprint:** V5.1 D1 common-B32 scorecard evidence, extended with the dedicated native inference-memory benchmark. Columns are reconstruction error, training time, training memory, inference time, and inference memory. Rows are ordered by reconstruction error.

## Evidence hierarchy

- **Hero evidence:** reconstruction error is the widest and first scorecard column; panel a establishes predictive-distribution quality.
- **Uncertainty validation:** panel b establishes state-level association; panel c adds the non-redundant operational selective-reconstruction consequence.
- **Resource qualification:** panel d reports each resource separately in real units and does not collapse them into a composite score.
- **SI/back-up:** C1a absolute selective risk and the former spatial error-capture panel remain outside the main figure.

## Statistics and display rules

- Panels a/b use the accepted V5 plotted samples and summaries in place; no bootstrap is rerun.
- Panel c uses the accepted 200-state, 64-draw C1b table and its 2,000-replicate, block-length-25 temporal moving-block-bootstrap intervals; rankings were recomputed inside each accepted resample.
- Panel d accuracy uses the frozen 1,000-state mean unobserved-field relative L2 and temporal-bootstrap 95% interval. Inference time uses accepted warm median and IQR. Common-B32 training time uses synchronized median update time and IQR; training memory uses peak allocated memory.
- Latent FM has two non-concurrent training stages. To preserve one aligned row and avoid reintroducing stage count, the main panel shows the larger stage value separately in each training column; both stage values and the fact that the maxima come from different stages are stated in the companion.
- Inference memory uses an open marker for unique parameters plus persistent buffers and a filled marker for process-local peak allocated memory during inference, joined by a thin line.

## Sources used in place

- Accepted V5 display source: `Dis_SI_Process/results/derived/20260831_1409/figure5_v5_source.csv`.
- V5.1 normalized selective-risk source: `Dis_SI_Process/results/ValidationV51/PanelC/panel_c_exploration_20260902_1129/selective_risk.csv`.
- V5.1 D1 common-B32 method and stage sources: `Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_plot_source_common_b32.csv` and `panel_d_stage_source_common_b32.csv`.
- Dedicated inference-memory source: `Dis_SI_Process/results/ValidationV51/InferenceMemory/inference_memory_native_v51_20260903_094021/inference_memory_summary.csv`.

## Integrity and reviewer risks

- All panels are quantitative; no representative raster image, crop, contrast adjustment, or pseudo-color operation is used. Dense accepted state/bootstrap clouds may be rasterized inside the SVG, while all text and summary marks remain editable vector elements.
- Finite empirical ensembles and formal underdispersion prevent a perfect-calibration or Bayesian-posterior claim. Positive Spearman association is informativeness, not calibration or prospective error prediction.
- C1b deliberately removes absolute between-method error differences; panel d retains absolute reconstruction error so the figure does not obscure baseline accuracy.
- Common-B32 is a common batch/precision/sensor protocol but retains method-native target workloads; resource coordinates are descriptive, hardware-specific checkpoint footprints, not a matched-budget causal efficiency ranking.
- Dedicated inference memory is process-local PyTorch allocated memory under `torch.inference_mode`; shared GPU work was explicitly allowed because no timing claim is attached to that benchmark.

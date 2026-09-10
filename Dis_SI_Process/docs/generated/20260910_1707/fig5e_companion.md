# Figure 5 V7R2 panel e: Scale-resolved fidelity

Scientific intent: Show population-spectrum shape and the distribution of selected-field high-band relative-L2 across all main variants plus Senseiver.

Quantitative definition: Median shell spectra are loaded from spectra_population.csv; source IQR values remain available for audit but are not plotted. High-band statewise relative-L2 and summary statistics are loaded from highband_states.csv and highband_summary.csv.

Evidence package: Package A, new ablation source reductions; last.pt only, with Truth and Senseiver reference rows.

Visual design: One lettered panel containing two vertically stacked axes; all 198 median shell points are connected, sparse markers every 42 shells identify methods, the high-band locator is shown on the spectrum, and the lower distribution plot retains all states.

The upper axis uses all 198 median shell points from spectra_population.csv; source IQR is retained in the reductions but is not plotted. Sparse markers every 42 shells identify methods and are presentation-only. The lower axis uses all selected-field high-band relative-L2 states and source summary rows from highband_states.csv/highband_summary.csv.

Source ledger entries from `source_manifest.json`:

| key | path | sha256 | role |
| --- | --- | --- | --- |
| highband_states.csv | Dis_SI_Process/results/derived/20260910_1707/highband_states.csv | 58b819c80e54001d0cd8e34669a76d25eaab3cf253a0011525dfa30624e0239c | Package A V7R2 saved-checkpoint reduction |
| highband_summary.csv | Dis_SI_Process/results/derived/20260910_1707/highband_summary.csv | acea4813cab3d3a9a2abdd4ef2217a93857fb2588e82cb3fb4c76f4ad0abb9f6 | Package A V7R2 saved-checkpoint reduction |
| spectra_population.csv | Dis_SI_Process/results/derived/20260910_1707/spectra_population.csv | ba3fb130125040c2e81aa1420b084ba02394d6d857041fe73b8f4969e0f65ad3 | Package A V7R2 saved-checkpoint reduction |
| spectral_diagnostics.csv | Dis_SI_Process/results/derived/20260910_1707/spectral_diagnostics.csv | 6ec66706008e20a52ff3e5f8c44f31f9571425b08e9d8aa655731ec57b04f731 | Package A V7R2 saved-checkpoint reduction |


Output contract: `fig5e_v7r2_20260910_1707.svg` and `fig5e_v7r2_20260910_1707.png`.

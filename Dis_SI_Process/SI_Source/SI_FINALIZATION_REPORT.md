# DMF-Gen Supplementary Information finalization report

## Deliverables

- Revised source: `CURRENT_Supplementary Info_FINAL.tex`
- Compiled PDF: `build_final/CURRENT_Supplementary Info_FINAL.pdf`
- Rebuilt figures: `figures_final/si01_mixed_resolution_distributions_final.pdf` through `si04_spatial_conditional_ensembles_final.pdf`
- Reproducible frozen-data renderer: `../figures/scripts/build_publication_ready_si_figures.py`
- Main-manuscript corrections: `MAIN_TEXT_SYNC.md`

The pre-existing SI source and PDF were not overwritten.

## Resolved inconsistencies

- Traced the mixed-resolution numerical path from raw HDF5 fields through preprocessing, selected channel index and frozen evaluation truth. The reconstructed variable is streamwise velocity \(V_x\). The frozen truth agrees with raw \(V_x\) and processed channel 0 to maximum absolute difference \(3.73\times10^{-9}\); a downstream density label is stale metadata.
- Standardized the SI to “training partition” and “held-out evaluation pool/cohort.” Each task now states when checkpoint selection and final evaluation share the held-out pool. Bootstrap intervals are explicitly limited to case/frame sampling under one selected trained realization.
- Defined the existing 1.00/0.34/0.44/0.16/0.19 quantity as trajectory-level native-grid spatial-DOF exposure, gave its formula once and stated that retained-frame differences, compute, update-sample count and total information are outside the metric.
- Defined the deterministic Fourier baseline as the Geo-FNO/RecFNO hybrid. The evaluated Cartesian path uses sparse value and support maps, and the implementation family also provides Geo-FNO-style geometry handling for non-tensor grids.
- Synchronized the Fig. 5f discussion with the final five coordinates: reconstruction error, training update time, training memory, inference time and inference memory. The fixed batch/workload, warm model-core boundary, inclusion/exclusion rules, memory definitions and Latent FM stage handling are explicit.
- Preserved the distinct approximately 0.117 and 0.1063 combustion analyses and made their checkpoint/cohort/estimator provenance explicit without assigning an unsupported causal explanation.
- Replaced internal checkpoint, cache and folder language with neutral scientific descriptions while retaining material protocol deviations. A second sentence-level editorial pass removed forensic proof text, implementation filenames and formulaic transitions from the SI body and captions.

## Figures and tables

- No reliable evaluated elasticity/airfoil reconstruction cache or finished irregular-domain SI asset was found. The unfinished figure and reference were removed; the four remaining supplementary figures were renumbered 1–4.
- Rebuilt all four retained figures from frozen CSV/NPZ/JSON evaluation artifacts. No checkpoint was loaded and no prediction inference was run. Method colors follow the final main-figure palette; all figure text is black, Arial is embedded, and panel lettering and notation are consistent.
- Supplementary Figure 1 now labels \(V_x\) and uses squared-energy allocation terminology. Supplementary Figures 2 and 3 state channel conditions, physical/error or spectral-scale conventions, cohort size, sensor count and estimator. Supplementary Figure 4 states that 256 temperature measurements and their locations remain fixed across 64 conditional draws within each state and that the displayed SD is across those draws.
- Added an early protocol summary table. Refined the analysis-specific combustion provenance table and consolidated comparison caveats into one implementation/comparison table.
- All tables use booktabs rules without vertical rules, repeat headers when continued and retain readable text size.

## Transparently stated limitations

- One trained realization is available per reported configuration; the reported intervals do not estimate training-seed variability.
- Held-out records are reused for checkpoint selection and final evaluation in the documented tasks.
- Architecture, frame windows, sensor placement and checkpoint-selection procedures differ in the identified comparison groups.
- The evaluated airfoil Geo-FNO/RecFNO checkpoint does not correspond to the lowest held-out loss in the complete log, and weights for the earlier logged minimum are unavailable.
- Existing evidence does not isolate how much of the approximately 0.117 versus 0.1063 difference is attributable to checkpoint weights versus the distinct evaluation pipelines.
- No direct cached-versus-recomputed airfoil sampler equivalence result was found; the SI reports the executed dependency structure without claiming such a test.

## Verification

- `latexmk` completed successfully: 22 pages, no undefined citations/references, no duplicate labels, no overfull/underfull boxes and no LaTeX warnings.
- Source and extracted-PDF text contain no `TBD`, `pending`, `placeholder`, `TODO`, author questions, revision-color commands, missing-figure markers or internal debugging language. The final prose was also checked for repeated sentences, formulaic transition patterns and cache/log narration.
- The source contains no full-line review comments and no blue/revision-only coloring. Every prose paragraph occupies one continuous source line.
- The final four figure PDFs embed Arial TrueType text and were visually checked at their final SI insertion size; the complete 22-page PDF was also inspected as a contact sheet and at full size on figure/table pages.
- No model training, checkpoint continuation or substantial new result generation was introduced.

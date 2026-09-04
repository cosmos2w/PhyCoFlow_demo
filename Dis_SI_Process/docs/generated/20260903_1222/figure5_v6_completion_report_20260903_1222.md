# Figure 5 V6 completion report

## Build status

- Starting commit: `f0aa4e7ca76bf15af2972fa434f295678ef1bfca`.
- Branch: `paper/postprocessing-multifield-superresolution`.
- Python renderer/export: complete.
- Structural and data QA: PASS.
- Print-size visual QA: PASS (Python-rendered previews inspected at 240 dpi).
- New scientific inference, bootstrap, training, or broad validation calculation: none.

## Files created

- Renderer: `Dis_SI_Process/figures/scripts/build_figure5_v6.py`.
- Contract/config: `Dis_SI_Process/docs/generated/20260903_1222/figure_contract.md` and `Dis_SI_Process/configs/figure5_v6.yaml`.
- Standalone SVGs: `Dis_SI_Process/figures/generated/20260903_1222/fig5a_probabilistic_reconstruction_20260903_1222.svg`, `Dis_SI_Process/figures/generated/20260903_1222/fig5b_uncertainty_tracks_difficult_states_20260903_1222.svg`, `Dis_SI_Process/figures/generated/20260903_1222/fig5c_selective_reconstruction_20260903_1222.svg`, `Dis_SI_Process/figures/generated/20260903_1222/fig5d_accuracy_computational_footprint_20260903_1222.svg`.
- Composed SVG: `Dis_SI_Process/figures/generated/20260903_1222/fig5_composed_v6_20260903_1222.svg`.
- Compact sources: `Dis_SI_Process/results/derived/20260903_1222/figure5_v6_source_index.csv` and `Dis_SI_Process/results/derived/20260903_1222/figure5_v6_panel_d_source.csv`.
- Manifest/QA: `Dis_SI_Process/results/derived/20260903_1222/build_manifest.json` and `Dis_SI_Process/results/derived/20260903_1222/qa.json`.
- One companion per standalone plus the composed companion under `Dis_SI_Process/docs/generated/20260903_1222`.

## Exact source reuse

- Panels a/b: accepted V5 display table `Dis_SI_Process/results/derived/20260831_1409/figure5_v5_source.csv`; all plotted state/bootstrap samples and summaries were reused without recalculation.
- Panel c: accepted V5.1 selective-risk table `Dis_SI_Process/results/ValidationV51/PanelC/panel_c_exploration_20260902_1129/selective_risk.csv`; only `risk_kind=normalized` (C1b) is plotted.
- Panel d accuracy/inference time: `Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_plot_source_common_b32.csv`; training time/memory: `Dis_SI_Process/results/ValidationV51/Derived/20260902_1129/panel_d_stage_source_common_b32.csv`; inference memory: `Dis_SI_Process/results/ValidationV51/InferenceMemory/inference_memory_native_v51_20260903_094021/inference_memory_summary.csv`.
- Source hashes and row counts are recorded in `Dis_SI_Process/results/derived/20260903_1222/figure5_v6_source_index.csv`.

## Final design choices

C1b replaces V5's spatial error-capture panel and V5.1's initially preferred C1a because it asks the distinct operational question: how well does each method's own uncertainty identify states suitable for selective retention? Dividing by each method's full-cohort error removes the already-represented absolute-accuracy difference, keeps every endpoint at one, and makes the panel a direct consequence of the spread–error association in b. C1a remains the absolute-error SI/back-up and was not copied or regenerated.

Panel d uses the D1 graphical-scorecard organization rather than lifecycle bubbles because reconstruction error is directly plotted in the first and widest column. Training and inference time/memory remain separate aligned real-unit quantities; stage count is removed. Dedicated inference-memory evidence supports the preferred open model-state / filled peak-allocated dumbbell. Latent FM uses per-column maxima of its non-concurrent stages to keep one row while the companion preserves both-stage interpretation.

## Dropped from V5/V5.1 main figure

- V5 spatial error capture, C1a, c2 interface profiles, c3 posterior atlas, and c4 functionals remain SI/back-up or internal evidence.
- D2–D5 candidate layouts, lifecycle bubbles/scatters, and stage count are excluded from the main figure.
- Development-style protocol text is removed from headers and moved to companions/caption.

## Limitations

Empirical ensembles are finite and underdispersed; spread/error evidence is informativeness rather than perfect calibration. C1b is normalized within method and must be read with the absolute accuracy evidence in d. Common-B32 training coordinates retain method-native target workloads, and all timing/memory results remain hardware/configuration-specific descriptive footprints. The dedicated inference-memory run allowed shared GPU use and makes no timing claim.

## Cleanup

No checkpoint, dataset, cache, old result bundle, ensemble stack, or repeated bootstrap array was copied. Temporary PNG previews were deleted after visual QA; no preview is retained.

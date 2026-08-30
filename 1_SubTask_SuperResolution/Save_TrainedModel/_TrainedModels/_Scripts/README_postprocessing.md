# Mixed-resolution paper post-processing

## Unified publication workflow v2

The additive `mixed_resolution_unified_v2` workflow reorganizes the main paper
figure into one a--f training-resolution study without changing the audited
formal cache, canonical index, sensor plan, or finalized result CSVs.

The main multiscale result uses a configurable orthogonal 2-D wavelet
decomposition of cached physical H-resolution fields. Panel d is the sensor
sweep, panel e shows representative large/intermediate/fine components, and
panel f quantifies scale-wise pattern correlation and variance-fraction bias.
The previous single M-resolution cutoff and FFT resolution-band analyses are
supplementary diagnostics; their CSVs remain under `UnifiedPublicationV2/` and
timestamped PDF copies are retained under
`_Process_Figures/MultiscaleWavelet/SI_Diagnostics_*/`.

```bash
conda run -n fig python 80_export_multiscale_wavelet.py \
  --run-id YYYYMMDD_HHMM \
  --cache-manifest ../_Process_Results/ReconstructionCache/ReconstructionCache_manifest_formal_20260712.csv \
  --canonical-index ../_Process_Results/CanonicalTestIndex/CanonicalTestIndex_formal_20260712.csv \
  --accuracy-summary ../_Process_Results/UnifiedPublicationV2/AllRecipeAccuracy_summary_<data-run>.csv
conda run -n fig python 81_plot_multiscale_components.py \
  --run-id YYYYMMDD_HHMM --data-run-id <data-run> --multiscale-run-id YYYYMMDD_HHMM \
  --cache-manifest ../_Process_Results/ReconstructionCache/ReconstructionCache_manifest_formal_20260712.csv
conda run -n fig python 82_plot_multiscale_fidelity.py \
  --run-id YYYYMMDD_HHMM --data-run-id <data-run> --multiscale-run-id YYYYMMDD_HHMM \
  --cache-manifest ../_Process_Results/ReconstructionCache/ReconstructionCache_manifest_formal_20260712.csv
```

```bash
conda run -n fig python 95_prepare_unified_v2_data.py \
  --run-id YYYYMMDD_HHMM --data-run-id 2026-07-13_14-55
conda run -n fig python 96_export_unified_v2_panels.py \
  --run-id YYYYMMDD_HHMM --data-run-id YYYYMMDD_HHMM
conda run -n fig python 97_assemble_mixed_resolution_unified_v2.py \
  --run-id YYYYMMDD_HHMM --data-run-id YYYYMMDD_HHMM
conda run -n fig python 98_audit_unified_v2.py \
  --data-run-id YYYYMMDD_HHMM
```

Preparation merges finalized accuracy and sweep summaries and derives missing
all-recipe coarse/detail statistics from existing cache arrays.  It does not
run model inference.  The optional `--allow-incremental-cache-fill` switch on
`95_prepare_unified_v2_data.py` is the only path that invokes cache generation;
it writes a separate `<run-id>_incremental` cache manifest on `cuda:2` and never
updates `ReconstructionCache_manifest_formal_20260712.csv`.

The composite defaults to qualitative Version 2 (three models and the three
central transfer recipes).  The standalone exporter also writes Version 1,
which uses two stacked strips to show all four models and four recipes without
forming an unreadable single row.

Run scripts from `1_SubTask_SuperResolution`. Every stage accepts `--run-id`; formal downstream stages prefer exact run-ID matches and only fall back to the latest artifact for interactive use without an ID.

The formal default is 300 distinct held-out CFD cases with one deterministic time per case, restricted to the intersection of all available run manifests' usable time windows. It uses H evaluation, nested 64/128/256/384/512/768/1024 H-grid sensors, 256 formal sensors, `last.pt`, two Euler steps, endpoint-smooth consistency where supported, and a configured CUDA device with CPU fallback. Missing model/recipe pairs are recorded rather than fatal.

`_Process_Results` must be a symlink to the cache disk by default. The cache preflight verifies the symlink, target free space, CUDA device, memory, and utilization. Compact caches store each float32 reconstruction and sensor indices once; H coordinates and physical truth are shared across all model/recipe/count entries and transparently rehydrated by analysis scripts.

Typical targeted workflow:

```bash
python Save_TrainedModel/_TrainedModels/_Scripts/00_inventory_models.py --run-id smoke
python Save_TrainedModel/_TrainedModels/_Scripts/01_build_sensor_plan.py --run-id smoke --snapshots 0 1
python Save_TrainedModel/_TrainedModels/_Scripts/02_build_reconstruction_cache.py --run-id smoke --models DMFGen Senseiver --recipes 2_H_limited 3_Mixed_HML --snapshots 0 1
python Save_TrainedModel/_TrainedModels/_Scripts/20_export_questionA_l2.py --run-id smoke
python Save_TrainedModel/_TrainedModels/_Scripts/21_plot_questionA_l2.py --run-id smoke
```

Formal resumable workflow:

```bash
python Save_TrainedModel/_TrainedModels/_Scripts/run_formal_postprocess.py --phase preflight --run-id YYYYMMDD_HHMM
python Save_TrainedModel/_TrainedModels/_Scripts/run_formal_postprocess.py --phase plan --run-id YYYYMMDD_HHMM
python Save_TrainedModel/_TrainedModels/_Scripts/run_formal_postprocess.py --phase cache_main --run-id YYYYMMDD_HHMM
python Save_TrainedModel/_TrainedModels/_Scripts/run_formal_postprocess.py --phase cache_sweep --run-id YYYYMMDD_HHMM
python Save_TrainedModel/_TrainedModels/_Scripts/run_formal_postprocess.py --phase export --run-id YYYYMMDD_HHMM
python Save_TrainedModel/_TrainedModels/_Scripts/run_formal_postprocess.py --phase plot --run-id YYYYMMDD_HHMM
python Save_TrainedModel/_TrainedModels/_Scripts/run_formal_postprocess.py --phase assemble --run-id YYYYMMDD_HHMM
```

All cache phases are resumable and flush the manifest every 25 entries. `91_assemble_mixed_resolution_publication.py` draws the 183 mm × 210 mm eight-panel figure directly from source CSV/cache evidence, retaining vector text, axes, contours, markers, and curves in SVG/PDF. Dense physical field layers are rasterized within the vector container. The composite and standalone exports share `common/publication_panels.py`; standalone a–h bundles are generated by `93_export_publication_panels.py` under `_Process_Figures/PublicationPanels/`. The corresponding `FigureSourceManifest_<run-id>.json` records source tables/caches, selected cases, truth-only ROI coordinates, normalization limits, rendering modes, missing references, layout proportions, and output paths.

`80_select_representative_snapshot.py` chooses median-like qualitative cases from paired quantitative errors; it does not select by visual appearance. After a completed run, rebuild/prune/audit with:

```bash
python Save_TrainedModel/_TrainedModels/_Scripts/03_rebuild_cache_manifest.py --run-id YYYYMMDD_HHMM
python Save_TrainedModel/_TrainedModels/_Scripts/04_prune_orphaned_cache.py --run-id YYYYMMDD_HHMM --apply
python Save_TrainedModel/_TrainedModels/_Scripts/94_audit_formal_workflow.py --run-id YYYYMMDD_HHMM
```

Plotters read CSV only. `30_export_comparison_contours.py` is the intentional exception because it renders cached spatial fields. `90_assemble_figure.py` reads existing PNG panels and never launches inference or recomputes analysis.

Use `--checkpoint best` only when explicitly desired. A missing requested checkpoint does not fall back unless `--allow-checkpoint-fallback` is supplied. Use `--paper` on plotters that expose it for the high-DPI SVG/PDF/PNG bundle.

The canonical fifth recipe is `5_ZeroH_MRich`, with paper label `Zero-H-M-rich` and ratio L:M:H = 1:2:0. The legacy downstream alias `5_ZeroH_LRich` is accepted only to migrate older derived artifacts; inventory preserves the actual trained directory name.

For an archive processed before this naming correction, run `05_migrate_zero_h_mrich_name.py` once. It renames derived folders and metadata, recomputes cache identities, and verifies every numerical cache array before and after migration without model inference.

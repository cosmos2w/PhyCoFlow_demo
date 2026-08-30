# Coupled-field reconstruction post-processing

This directory implements the paper workflow for “Coupled-field reconstruction from incomplete measurement channels.” It is evaluation-only: no training file or training behavior is modified.

## Design

The workflow has four strict layers:

1. `00_inventory_models.py` records every expected model-condition pair, including missing placeholders.
2. `01_build_sensor_plan.py` creates one deterministic point-index plan per condition and test snapshot.
3. `02_build_reconstruction_cache.py` is the only post-processing script that loads models. It delegates architecture construction and inference to `src/evaluate_coherence.py`, `src/model_baseline.py`, and their existing adapters.
4. Exporters read caches; dedicated plotters read CSV only. Plotters never load checkpoints.

PointCloudFFM models request `endpoint_smooth` by default. Models whose native adapters do not implement it use their native conditional inference and record `native_not_applied` in cache/metric metadata.

## Quick smoke workflow

Run from `_Scripts/` in a Python environment containing both the trained-model dependencies and matplotlib. In this workspace, `phycoflow_env` satisfies that combined requirement (`fig` has matplotlib but currently lacks PyTorch):

```bash
conda run --no-capture-output -n phycoflow_env python run_postprocess.py --mode smoke --run-id smoke_20260711
```

This processes at most two snapshots, then runs contours, Field-L2, representative PDFs, JSD, all plotters, and the example assembler. It continues across missing model groups. A load/inference failure becomes a status row and Missing panel.

The orchestrator exits if an individual script itself has a programming/configuration error; model-specific failures are caught inside the inventory/cache workflow.

## Full test-set command (not run automatically)

```bash
conda run --no-capture-output -n phycoflow_env python run_postprocess.py --mode full --run-id paper_v1
```

For more control, run stages individually:

```bash
python 00_inventory_models.py --run-id paper_v1 --checkpoint last --probe-load
python 01_build_sensor_plan.py --run-id paper_v1 --split test
python 02_build_reconstruction_cache.py --run-id paper_v1 \
  --sensor-plan ../_Process_Results/SensorPlans/SensorPlan_paper_v1.csv \
  --checkpoint last --n-steps 2 --ode-solver euler --obs-consistency endpoint_smooth
python 20_export_field_l2.py --run-id paper_v1 \
  --cache-manifest ../_Process_Results/ReconstructionCache/ReconstructionCache_manifest_paper_v1.csv
python 21_plot_field_l2_heatmap.py --run-id paper_v1 --scale linear --formats png pdf svg
python 50_export_energy_spectra.py --run-id paper_v1 \
  --cache-manifest ../_Process_Results/ReconstructionCache/ReconstructionCache_manifest_paper_v1.csv
python 51_plot_energy_spectra.py --run-id paper_v1 --formats png pdf svg
python 52_export_spectral_lsd.py --run-id paper_v1 \
  --cache-manifest ../_Process_Results/ReconstructionCache/ReconstructionCache_manifest_paper_v1.csv
python 53_plot_spectral_lsd.py --run-id paper_v1 --condition Cond_T
python 54_plot_spectral_validation_composite.py --run-id paper_v1 --condition Cond_T
```

Use `--checkpoint best` explicitly for best checkpoints. A missing `last.pt` is not replaced with `best.pt` unless `--allow-checkpoint-fallback` is supplied.

## Configuration and aliases

Edit `postprocess_config.yaml` to change directory aliases, method order/palette, conditions, fields, colormaps, sizes, typography, line widths, missing-data appearance, bins, seeds, DPI, or export formats. Family detection does not rely on these directory aliases: `run_config.yaml` and checkpoint metadata are passed to the same inference logic used by the canonical coherence evaluator.

Default conditions use canonical field indices `CH4=0`, `CO=1`, `T=2`, `U1=3`, and `p=4`:

- `Cond_T`: field 2, 256 sensors.
- `Cond_TU1`: fields 2 and 3, 256 sensors each.
- `Cond_COTU1P`: fields 1, 2, 3, and 4, 256 sensors each.

These requested mappings override copied visualization fields in training configs.

## Artifact schemas

`ModelInventory_<run-id>.csv` has one row for all 8 × 3 expected pairs. Status values are `ok`, `missing directory`, `missing config`, `missing checkpoint`, `missing dependency`, `load error`, or `inference error`.

`SensorPlan_<run-id>.csv` is long-form: split, condition, snapshot, deterministic sensor seed, sensor order, channel, point index, and normalized observed value. Sensor selection is regenerated and verified against canonical baseline selection before inference results are accepted.

Each compressed reconstruction cache contains:

- normalized and physical truth/reconstruction;
- normalized and physical coordinates;
- observation point indices, fields, and normalized values;
- JSON metadata with model family/backbone, exact checkpoint and modification time, checkpoint policy, sensor-plan hash, split/snapshot, sensor and generation seeds, NFE, solver, requested/applied consistency policy, and cache identity.

The identity changes when checkpoint, checkpoint mtime, sensor plan, split, snapshot, flow steps, solver, consistency policy, or generation seed changes.

Long full-test cache builds are resumable. Reuse the same run ID and sensor-plan
CSV; existing NPZ entries are retained, and the manifest is flushed every 25
snapshots by default. If a process is interrupted after writing cache files,
rebuild its manifest without loading a model:

```bash
python 02_build_reconstruction_cache.py --run-id paper_v1 --sensor-plan ../_Process_Results/SensorPlans/SensorPlan_paper_v1.csv --manifest-flush-every 25
python 03_rebuild_cache_manifest.py --run-id paper_v1
```

When the requested run ID, sensor-plan hash, checkpoint kind, flow settings,
and snapshot scope match a complete manifest, `02_build_reconstruction_cache.py`
now prints `[FOUND] existing reconstruction cache` and reuses the recorded NPZ
entries **without loading the model**. This also retains prior explicit missing
or load-error placeholders. To intentionally replace an existing cache, pass
`--force-regenerate` (the legacy `--force` alias remains supported):

```bash
python 02_build_reconstruction_cache.py --run-id paper_full_20260711 \
  --sensor-plan ../_Process_Results/SensorPlans/SensorPlan_paper_full_20260711.csv
python 02_build_reconstruction_cache.py --run-id paper_full_20260711 \
  --sensor-plan ../_Process_Results/SensorPlans/SensorPlan_paper_full_20260711.csv \
  --force-regenerate

For the paper cache, `postprocess_config.yaml` applies `default_hard`
observation consistency to **DMF-Gen only** through
`method_inference_overrides`. This replaces the older DMF-Gen
`endpoint_smooth` cache policy while leaving the global default available to
the other point-cloud models. Pass `--obs-consistency ...` to
`02_build_reconstruction_cache.py` only when intentionally overriding every
selected method.
```

`run_postprocess.py` forwards the same `--force-regenerate` option. With the
default behavior, it automatically detects and reuses the existing cache.

### Family-specific paper sampling settings

The shared default is two flow steps, but `method_inference_overrides` in
`postprocess_config.yaml` records the paper-specific exception for SiT:
`SiT: {n_steps: 4}`. This reproduces the canonical SiT evaluation setting while
retaining the requested `last.pt` checkpoint policy. An explicit
`02_build_reconstruction_cache.py --n-steps N` intentionally overrides this
per-method setting for every selected model and is recorded in cache metadata.

`FieldL2_per_snapshot_<run-id>.csv` stores all-point physical relative L2, physical relative L2 after removing direct observed point-channel entries, normalized relative L2, sensor consistency, observed flag/count, inference metadata, and status. `Unobserved_mean` is an additional row per method-condition-snapshot. `FieldL2_summary_<run-id>.csv` stores mean, sample standard deviation, median, quartiles, snapshot-bootstrap 95% CI, valid count, and status.

The sensor plan defines expected test coverage. A summary receives status `ok`
only when every planned snapshot for that model-condition is valid; a partly
written cache is explicitly marked `incomplete cache`. This prevents a resumed
run from being mistaken for a complete all-test-set result.

`JointPDF_snapshot_<run-id>.csv` stores long-form physical bin rectangles and probabilities for one shared truth reference plus each requested reconstruction; truth is never duplicated for every method. Its metrics CSV stores representative base-2 JSD. `31_plot_joint_pdf_snapshot.py --tag all` preserves a full-model contact sheet, while `--subset ... --tag main` writes a compact publication subset without overwriting it. `JointPDF_JSD_per_snapshot_<run-id>.csv` and its summary use fixed robust global truth-derived edges and the configured common pseudocount.

### Channel-coupling JSD panels

`42_export_coupling_jsd.py --run-id <run-id>` creates the publication-oriented
`CouplingJSD_per_snapshot_<run-id>.csv` and summary for T--U1 (thermal--flow),
CO--U1 (chemistry--flow), and CO--T (thermal--chemistry).  It never loads a
model or regenerates a reconstruction: finalized T--U1 and T--CO values are
reused (CO--T is the mathematically equivalent transposed orientation), while
only CO--U1 is calculated from the existing cache.  Render the richer
conditioning regimes for supplementary information with:

```bash
python 55_plot_coupling_jsd_si.py --run-id <run-id>
```

This writes one CSV-only three-coupling panel per condition under
`_Process_Figures/JointPDF_JSD/Supplementary/`.

The main-manuscript and SI placement of the nine available coupling plots is
controlled at the header of `91_assemble_coupled_field_publication.py` through
`PANEL_D_FIGURE_SELECTIONS`.  Every entry is an editable `{"pair": ..., "condition": ...}` record.  Render the main selection alone with:

```bash
python 91_assemble_coupled_field_publication.py --run-id <run-id> --panel d --output-id panel_d_check
```

Render the two header-defined SI selections with:

```bash
python 55_plot_coupling_jsd_si.py --run-id <run-id> --figures si_1 si_2 --output-id panel_d_si_check
```

### Channel-wise spectral validation

Spectral analysis is cache-only and uses denormalized physical-unit channel fields. It does not describe arbitrary channels as kinetic-energy spectra: figures and tables use **channel-wise spectral energy** or **power spectrum**.

`EnergySpectra_snapshot_<run-id>.csv` contains a long-form curve with one ground-truth record per condition/snapshot/channel and one reconstruction record per available model. `EnergySpectra_snapshot_metrics_<run-id>.csv` reports representative dB/natural-log LSD, total-energy ratio, and low/mid/high-band ratios/errors. Its metadata records the structured-grid decision and preprocessing options.

`SpectralLSD_per_snapshot_<run-id>.csv` stores dB LSD, natural-log LSD, energy ratios, cache/inference metadata, coordinate mode, window, and status for every cached model-condition-snapshot-channel. `SpectralLSD_summary_<run-id>.csv` contains the snapshot mean, standard deviation, median, quartiles, and bootstrap 95% CI for dB LSD, plus natural-log LSD and total-energy-ratio summaries.

Interpret spectral LSD as a comparison of shell-averaged spectral-energy magnitudes, not a spatial-image fidelity score: phase and the location of structures are intentionally discarded by radial averaging. A method can therefore have low LSD/JSD while retaining a relatively high pointwise L2 error; publication plots retain both quantities rather than altering either metric to force agreement.

For full-test-set LSD exports, channels from the same cache entry are processed as one batch. `spectral.compute_device: auto` uses CUDA when available; this moves the double-precision FFT and shell reductions to the selected GPU while NPZ loading remains CPU/I/O bound. Use `--device cuda:1` to select a GPU or `--device cpu` for the reference backend. The exported metadata records the resolved device. GPU and CPU values are numerically verified against the same native-shell definition.

The primary paper metric is

```text
lsd_db = sqrt(mean([10 log10((E_pred(k)+eps)/(E_true(k)+eps))]^2))
```

where `eps = max(1e-30, relative_epsilon * max(E_true))`. The compatibility metric is the canonical evaluator's natural-log form,

```text
lsd_loge = sqrt(mean([log(E_pred(k)+eps) - log(E_true(k)+eps)]^2)).
```

The utility reconstructs a complete 2D grid only when explicit `Num_x`/`Num_y` metadata is valid or the coordinate product is complete. It never interpolates arbitrary point clouds. In `coordinate_mode: auto`, approximately uniform physical spacing yields physical wavenumbers; otherwise it uses a unit-spaced topological/index grid and records that choice. FFT spectra remove the spatial mean by default, optionally apply a Hann window with energy correction, omit the zero shell, use native shell spacing, and by default retain only the isotropically resolved range. Cond_T is the recommended main-paper condition; all three conditions are supported for supplementary analyses.

## Figures

```bash
python 10_export_contours.py --run-id paper_v1 --models DMFGen --conditions all --snapshot 0
python 31_plot_joint_pdf_snapshot.py --run-id paper_v1 --models all --tag all
python 31_plot_joint_pdf_snapshot.py --run-id paper_v1 --subset DMFGen SiT GeoFNO Senseiver --tag main
python 41_plot_joint_pdf_jsd_violin.py --run-id paper_v1 --jitter
python 50_export_energy_spectra.py --run-id paper_v1 --condition Cond_T --snapshot-index 0
python 51_plot_energy_spectra.py --run-id paper_v1 --reduced-models DMFGen SiT GeoFNO Senseiver
python 52_export_spectral_lsd.py --run-id paper_v1 --conditions Cond_T Cond_TU1 Cond_COTU1P --device cuda:1
python 53_plot_spectral_lsd.py --run-id paper_v1 --condition all
python 54_plot_spectral_validation_composite.py --run-id paper_v1 --condition Cond_T --channels CH4 CO U1
python plot_condition_matrix.py --run-id paper_v1 --formats png pdf svg
python 90_assemble_figure.py --run-id paper_v1 --layout example_layout.yaml --formats png pdf svg
python 90_assemble_figure.py --run-id paper_v1 --layout publication_figure_layout.yaml --formats png pdf svg --vector-pdf
```

Contours use physical coordinates/values, triangulation, common per-field value limits, pooled robust error limits with `extend=max`, field-specific sensor overlays, observed/unobserved labels, and per-field physical relative L2. Each GT, reconstruction, and error plot is exported as an individual physical-field panel named `Fig_<type>_<field>_s<snapshot>_<model>_<condition>_<run-id>`. The map is the primary panel; its adjacent colorbar matches its height. Contours default to 50 levels, error colorbars show exactly four scientific-notation ticks, and all displayed numeric values use three significant figures in scientific notation. Missing reconstructions and errors are grey panels. `--y-compression` defaults to 1.0.

The assembler does not rerun analysis. Its YAML supports rows, columns, spans, explicit panel labels, relative/absolute panel paths, `{run_id}` substitution, dimensions, gutters, and Missing text. It also supports `type: contour_grid`, which resolves the standalone contour contract without manually listing paths:

```yaml
type: contour_grid
model: DMFGen
condition: Cond_T
snapshot: 0
fields: [T]              # or [CH4, CO, T, U1, p]
kinds: [GT, Rec, Err]
contour_run_id: "{run_id}"
```

Each requested field/type becomes one nested image tile; unavailable files remain visible as Missing tiles. Raster input panels can be embedded in editable SVG/PDF assemblies; assemble SVG elements externally if fully vector-preserved internals are required.

`publication_figure_layout.yaml` is the curated five-block double-column layout: three CH4 reconstruction regimes, all-method Field-L2 heatmaps, spectral validation, representative Cond_T joint PDFs, and all-condition JSD violins. Its `--vector-pdf` mode composes PDF source panels through LaTeX/TikZ so vector source graphics remain vector in the final PDF.

## Reproducibility and limitations

- Formal metrics use physical units; normalized arrays remain in caches for diagnostics.
- GPU memory is released after every method-condition group.
- A cache represents one reproducible generation draw per model/snapshot. Multiple stochastic repeats require distinct run IDs/seeds and are not pooled automatically.
- The fixed sensor plan assumes identical test split, point ordering, and data preprocessing across methods. Baseline inference is rejected if canonical selection does not reproduce the plan.
- A portable Latent-FM archive may store its shared VAE in `Latent_FM/Stage0/`. When a stage-2 checkpoint records an obsolete external VAE path, the loader first validates the recorded path and then uses that local archive artifact (preferring `last.pt`, then `best.pt`). If neither the recorded dependency nor a local `Stage0`/legacy `Stage1` checkpoint exists, it is reported as `missing dependency`.
- Empty method directories remain visible as NaN/status rows and Missing panels.
- SVG is always emitted as the primary editable-text figure even when YAML lists PNG only; PNG defaults to 150 DPI, and `--dpi 600` supplies a publication raster.

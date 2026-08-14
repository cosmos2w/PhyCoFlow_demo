# Canonical dataset-generation plan

## Scope and benchmark roles

The four notebook demonstrations will become dense, simulation-grounded source datasets for later sparse-reconstruction benchmarks.

| Case | Coherence role | Canonical ML fields | Solver-native state |
|---|---|---|---|
| 1D viscous Burgers | single field | `u` | `u` |
| 1D Kuramoto--Sivashinsky (KS) | single field | `u` | `u` |
| 2D Brusselator reaction--diffusion | multiple coupled fields | `u, v` | `u, v` |
| 2D Kolmogorov flow | multiple coupled fields | `u, v, p` | vorticity `omega` |

Sparse observation masks should not be baked into these canonical dense simulations. They should be generated later from split-aware manifests so that different reconstruction models see exactly the same ground truth and observation patterns.

## Readiness assessment (2026-08-14)

### What can be run now

All four notebooks execute successfully with the current base Jupyter kernel. Fresh smoke-mode executions produced finite fields and the following short-run diagnostics:

| Case | Smoke result |
|---|---|
| Burgers | relative PDE residual `7.98e-6`; mean drift `1.95e-17` |
| KS | relative PDE residual `3.09e-3`; finite output |
| Brusselator | residuals `6.95e-5` (`u`) and `2.38e-5` (`v`); positive concentrations |
| Kolmogorov flow | momentum residuals `1.53e-5`, `2.16e-5`; divergence RMS `1.47e-15`; pressure mean `1.04e-17` |

The notebooks are suitable demonstrations, but they are not yet suitable as formal dataset jobs:

1. Parameters, seeds, paths, and resolutions are hard-coded.
2. Each notebook holds all snapshots in memory and does not provide atomic, resumable trajectory output.
3. They use NumPy FFTs and therefore run on CPU only.
4. Their only persistent artifact is the executed notebook; there is no standardized raw-data or metadata contract.
5. The requested 2D resolution is 192 x 192, while both 2D notebooks currently use 64 x 64.
6. One fixed notebook run is not a dataset. Multiple independently seeded initial conditions are required, with case-based train/validation/test splits.

Therefore: the original notebook settings can be run directly, but formal 192 x 192 dataset generation should wait for the scripted, validated implementation below.

### GPU feasibility

The host has three NVIDIA RTX 6000 Ada Generation GPUs with 48 GB memory each. At inspection time, GPU 1 was idle and GPUs 0 and 2 were occupied. The `phycoflow_env` environment contains PyTorch `2.5.1+cu121`, reports CUDA available, and also contains NumPy, SciPy, h5py, and Matplotlib. CuPy and JAX are not installed.

The implementation should use a small array-backend abstraction with:

- `numpy` as the trusted CPU reference backend;
- `torch` plus `torch.fft` as the GPU backend;
- explicit `--device`, `--solver-dtype`, and `--batch-trajectories` options;
- identical equations, FFT conventions, de-aliasing masks, seeds, and output layout on both backends.

An FFT-only microbenchmark at shape `[8, 192, 192]` measured approximately `0.083 ms` per CUDA float32 FFT and `0.131 ms` per CUDA float64 FFT, compared with `4.53 ms` and `9.98 ms` on an eight-thread CPU run. This is not an end-to-end solver benchmark, but it confirms that batched 192 x 192 spectral trajectories can benefit substantially from the GPU. Small 1D jobs may remain CPU-efficient because GPU launch and transfer overhead can dominate.

Use physical GPU assignment outside the program and logical device assignment inside it:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n phycoflow_env \
  python datagen/launch_generation.py --case brusselator --device cuda:0 ...
```

Do not silently fall back from CUDA to CPU. The launcher should print the physical visibility mask, logical device, GPU name, backend, dtypes, estimated storage, and full resolved configuration before starting.

For canonical data, validate float32/complex64 GPU trajectories against a float64/complex128 CPU reference over a short non-chaotic horizon. If convergence or balance diagnostics are not adequate, use float64 on the GPU or reduce the time step. Long chaotic trajectories are expected to diverge pointwise across precision/backends; their validation must use conserved quantities, residuals, spectra, and stationary statistics rather than final-frame equality.

### 192 x 192 numerical caution

Raising `n` from 64 to 192 is straightforward for the spectral operators but still requires a numerical qualification run. Do not assume that a finite trajectory is resolved or accurate. Before production:

- compare 192 x 192 against at least one 256 x 256 reference trajectory from the same initial condition;
- compare the notebook time step with a half-step run;
- inspect the high-wavenumber spectral tail and the 2/3 de-alias cutoff;
- check PDE residuals and case-specific physical balances;
- ensure 192-grid results are not merely interpolated 64-grid results.

The 2D default resolution will be 192 x 192. The 1D defaults remain 512 points for Burgers and 256 points for KS.

### Large-data location and symbolic link

`/data/wanglz` is writable by the current user. It has about 233 GB available, while the project filesystem has only about 39 GB available. A symbolic link is therefore appropriate, but `/data` is already 94% full and production jobs need a preflight free-space check plus a safety margin.

Proposed real and linked paths:

```text
/data/wanglz/PhyCoFlow/datagen/       # real storage
<repository>/datagen/data -> /data/wanglz/PhyCoFlow/datagen
```

Implement `datagen/setup_data_root.py` to create the target and link safely. It must refuse to replace an existing non-symlink path, verify write access, resolve and print both paths, and support `--check-only`. The data root must remain outside version control.

At float32, one processed trajectory at notebook cadence is approximately:

| Case | Frames and shape | Processed fields per trajectory |
|---|---|---:|
| Burgers | `201 x 1 x 512` | 0.39 MiB |
| KS | `401 x 1 x 256` | 0.39 MiB |
| Brusselator | `241 x 2 x 192 x 192` | 67.8 MiB |
| Kolmogorov flow | `201 x 3 x 192 x 192` | 84.8 MiB |

The Kolmogorov raw vorticity adds about 28.3 MiB per trajectory; other raw state sizes are close to their processed field sizes. Keeping both uncompressed raw state and processed fields for 100 trajectories of every case is roughly 24 GB before metadata and filesystem overhead. The storage estimator must use actual requested sizes, include temporary-file and HDF5 overhead, reserve at least 15% free space, and refuse unsafe launches unless the user explicitly overrides the check.

## Canonical parameter defaults from the notebooks

| Setting | Burgers | KS | Brusselator | Kolmogorov flow |
|---|---:|---:|---:|---:|
| Spatial resolution | 512 | 256 | **192 x 192** | **192 x 192** |
| Domain | `[0, 2pi)` | `[0, 60)` | `[0, 20)^2` | `[0, 2pi)^2` |
| Solver `dt` | 0.0025 | 0.05 | 0.01 | 0.01 |
| Burn-in | 0 | 50 | 0 | 20 |
| Recorded time | 2 | 100 | 12 | 10 |
| Save cadence | 0.01 | 0.25 | 0.05 | 0.05 |
| Physical parameters | `nu=0.01` | canonical unit coefficients | `A=1, B=3, Du=1, Dv=0.1` | `Re=40, forcing=sin(4y)e_x` |

The first canonical release should keep these physical parameters fixed and vary only controlled smooth initial conditions. Parameter sweeps should be separate named dataset variants, with every varying physical parameter recorded in `conditions`. This prevents an ambiguous mixture of initial-condition diversity and equation diversity.

Burgers currently has no random input, so its script must define a documented seeded initial-condition family, for example bounded perturbations to the amplitudes and phases of the notebook's three Fourier modes. The notebook initial condition remains trajectory zero/reference. KS, Brusselator, and Kolmogorov flow already have seeded smooth-noise constructions; expose their amplitude and spectral-filter parameters.

## Proposed implementation layout

```text
datagen/
  DATASET_GENERATION_PLAN.md
  setup_data_root.py
  estimate_storage.py
  launch_generation.py
  process_to_h5.py
  validate_dataset.py
  common/
    backend.py                 # NumPy/Torch device and dtype handling
    config.py                  # dataclasses, CLI parsing, resolved-config export
    spectral.py                # FFT grids, derivatives, de-aliasing, ETDRK4 helpers
    raw_io.py                  # atomic/resumable per-trajectory NPZ + JSON
    h5_schema.py               # schema creation and streaming writes
    diagnostics.py             # shared residual/statistical utilities
  1_burgers/
    generate.py
    visualize.py
  2_ks/
    generate.py
    visualize.py
  3_brusselator/
    generate.py
    visualize.py
  4_navier_stokes/
    generate.py
    visualize.py
  tests/
    test_smoke_all_cases.py
    test_cpu_gpu_equivalence.py
    test_h5_schema.py
    test_reproducibility.py
```

Keep the notebooks as explanatory references. Move reusable numerical logic into Python modules; notebooks should eventually import those modules rather than maintain a second solver implementation.

## Generation CLI contract

`launch_generation.py` will route to a case implementation, schedule trajectory batches on one explicitly assigned device, and save the fully resolved run configuration. Every case script must also remain directly callable for debugging.

Common options:

```text
--case
--output-root
--dataset-id
--num-trajectories
--seed-start / --seeds-file
--resolution (one integer for the current periodic square cases)
--dt
--burn-in-time
--record-time
--save-every
--backend {numpy,torch}
--device {cpu,cuda:0,...}
--solver-dtype {float32,float64}
--storage-dtype {float32,float64}
--batch-trajectories
--checkpoint-every
--resume
--dry-run
--num-workers
```

Case-specific physical options:

- Burgers: `--viscosity`, `--domain-length`, mode amplitudes/phases, perturbation bounds.
- KS: `--domain-length`, coefficients if a noncanonical variant is intentionally requested, initial noise amplitude/filter.
- Brusselator: `--A`, `--B`, `--diffusivity-u`, `--diffusivity-v`, `--domain-length`, initial noise amplitude/filter.
- Kolmogorov flow: `--reynolds-number` or `--viscosity` (mutually exclusive), `--forcing-amplitude`, `--forcing-wavenumber`, `--domain-length`, perturbation amplitude/filter, ETDRK contour points.

All arguments need units, defaults, equations, and stability implications in comments/help text. Validate integral step counts and require `save_every` to be a positive integer number of solver steps. Store both step indices and physical times to avoid floating-point ambiguity.

Each trajectory must be independently reproducible from `(schema version, case, resolved config, seed, code commit)`. A failed job must leave an identifiable temporary file, never a valid-looking final trajectory. `--resume` should skip only trajectories that pass checksum/shape/metadata validation.

## Standardized raw-data contract

Use one uncompressed NPZ per trajectory for simple atomic writes, independent retries, and `allow_pickle=False` loading. Large arrays should not be collected into one monolithic NPZ. A dataset-level manifest and per-trajectory JSON carry metadata that should not be encoded as pickled Python objects.

```text
data/raw/<case>/<dataset_id>/
  manifest.json
  resolved_config.json
  trajectories/
    trajectory_000000.npz
    trajectory_000000.json
    trajectory_000001.npz
    trajectory_000001.json
  logs/
  checksums.sha256
```

Required NPZ arrays:

- `state`: `[time, state_channel, *spatial]`, solver-native saved state;
- `time`: `[time]`, float64 physical time after burn-in;
- `step`: `[time]`, int64 solver step relative to recording start;
- `x`: `[nx]`, float64 physical periodic coordinates;
- `y`: `[ny]` for 2D only.

Required JSON content includes case name, trajectory ID, seed, state-channel names, equation and physical parameters, domain/boundary conditions, resolution, solver/integrator, time controls, backend/device, solver/storage dtypes, de-aliasing rule, code commit, package versions, timestamps, array shapes, and diagnostic summary. The JSON must distinguish burn-in time from recorded time.

Kolmogorov raw state is vorticity. Its postprocessor reconstructs velocity, recovers pressure from the periodic Poisson equation, and fixes the pressure gauge to zero spatial mean. This derivation and sign convention must be tested explicitly.

## Unified processed HDF5 contract

Match the existing project loader's tensor convention so the new cases can reuse downstream data code:

```text
fields       [B, T, N, 1, 1, C] float32
coordinates  [N, 1, 1, 3]       float32
time         [T]                 float64
conditions   [B, P]              float32
```

Here `N=nx` for 1D and `N=ny*nx` for 2D, flattened in documented C order with `x` varying fastest. Unused coordinate dimensions are zero. `C` is 1 for Burgers/KS, 2 for Brusselator, and 3 for Kolmogorov flow. Coordinates remain in physical units; model-specific coordinate normalization happens in the loader. Fields remain in physical units; normalization statistics are derived from training trajectories only.

Add the following schema elements without breaking the four core datasets:

```text
trajectory_id              [B] UTF-8
seed                       [B] int64
splits/train               [B_train] int64 trajectory indices
splits/validation          [B_val] int64 trajectory indices
splits/test                [B_test] int64 trajectory indices
statistics/train_mean      [C] float64
statistics/train_std       [C] float64
diagnostics/<metric>       [B] or [B,T]
auxiliary/vorticity        optional, Kolmogorov flow only
```

Required attributes include `schema_version`, `case_name`, `equation`, `field_names`, `field_units`, `condition_names`, `state_names`, `spatial_dimension`, `grid_shape`, `domain_bounds`, `periodic_axes`, `solver`, `integrator`, `dealiasing`, `pressure_gauge`, `source_dataset_id`, `code_commit`, and creation/package provenance. Put structured metadata in a UTF-8 JSON dataset when it does not fit cleanly into HDF5 scalar attributes.

Use snapshot-oriented chunks such as `(1, 1, N, 1, 1, C)`, configurable `gzip`/`lzf` compression, checksums where supported, and streaming trajectory writes so the whole dataset is never resident in RAM. Write to `dataset.h5.tmp`, validate it, then atomically rename it. Never overwrite an existing complete dataset unless an explicit, separately confirmed option is supplied.

`conditions` contains only physical parameters that vary within that dataset variant. Seeds and trajectory IDs have dedicated datasets. If no physical parameter varies, use shape `[B, 0]`, not the existing legacy placeholder shape `[1, 0]`.

Splits are by trajectory, never by frame, to prevent temporal leakage. Use a deterministic seed and a documented default ratio such as 80/10/10. A parameter-generalization benchmark should use a separate explicit split policy rather than silently mixing it with the initial-condition benchmark.

## Postprocessing workflow

`process_to_h5.py` should:

1. Read and validate the raw manifest and every selected trajectory without pickle.
2. Verify identical grids, saved times, channels, and fixed parameters within one output file.
3. Derive canonical fields from native state; for Kolmogorov flow derive `u, v, p` from `omega`.
4. Stream-cast to storage dtype and flatten space into the existing loader layout.
5. Compute case-based splits before normalization statistics.
6. Compute training-only per-channel statistics with an online float64 algorithm.
7. Compute/store diagnostics and optional auxiliary quantities.
8. Write provenance, checksums, and a human-readable summary.
9. Run the schema/physics validator and only then atomically publish the HDF5 file.

Do not downsample 192 x 192 output to manufacture the canonical high-resolution target. If later super-resolution variants are needed, derive lower resolutions spectrally (preferred for periodic data) under a separate script and record the exact filter/downsampling operator.

## Visualization and physical-quality scripts

Each case gets a small `visualize.py` adapter built on shared plotting/diagnostic utilities. It must read either raw trajectory NPZ or processed HDF5, select `--trajectory`, `--time-index`/`--time`, and save reproducible static figures. Optional animation output should be explicit because notebook JavaScript animations can be large.

Common plots:

- field snapshot(s) and space--time view where appropriate;
- marginal value distributions;
- per-field 1D or radial spectra with the de-alias cutoff shown;
- joint field-value point clouds for multi-field cases;
- threshold/level-set geometry;
- time histories of residuals and physical diagnostics;
- optional raw-versus-HDF5 round-trip comparison.

Case-specific checks, preserving and extending the notebooks:

- Burgers: shock gradient, mean/mass drift, energy decay and viscous dissipation, PDE residual.
- KS: space--time chaos, spatial phase portrait, mean drift, spectrum, temporal/spatial variance, PDE residual and post-burn stationarity.
- Brusselator: `u/v` maps, joint cloud, marginal/radial spectra, median contours, positivity, per-equation residuals, cross-correlation and pattern wavelength.
- Kolmogorov flow: `u/v/p/omega`, joint cloud, kinetic-energy spectrum, vorticity contours, divergence, pressure-gauge error, momentum residuals, energy/enstrophy, forcing injection and viscous dissipation.

Use consistent axis conventions (`fields[..., y, x]`, origin lower), physical extents, channel names, units, color normalization, and output metadata. A `--report` mode should create a compact QA directory with PNG/PDF plots plus `diagnostics.json`, not modify the source notebooks.

## Validation gates before production

### Gate 1: unit and smoke tests

- FFT derivative tests on analytic periodic functions.
- ETDRK4 coefficient and one-step regression tests.
- pressure reconstruction sign/gauge test.
- raw NPZ and HDF5 shape/metadata round trip.
- one tiny trajectory for each case on CPU and CUDA.

### Gate 2: backend and reproducibility tests

- repeat a trajectory with the same seed/backend/config;
- short-horizon CPU float64 versus GPU float64 comparison;
- short-horizon GPU float32 versus float64 error and diagnostics;
- verify different seeds produce nonduplicate initial conditions and trajectories;
- save the exact resolved configuration and code commit.

### Gate 3: discretization qualification

- 192 versus 256 resolution comparison for both 2D cases;
- `dt` versus `dt/2` comparison;
- spectrum-tail/de-alias inspection;
- residual and physical-balance thresholds selected from the converged runs, not copied from smoke mode;
- confirm burn-in stationarity for KS and Kolmogorov flow.

### Gate 4: pilot dataset

Generate 3--5 trajectories per case, postprocess, validate, visualize, and load samples through the existing PyTorch dataset convention. Measure end-to-end wall time, peak GPU/RAM use, compression ratio, and actual bytes per trajectory. Use these measurements to set safe batch sizes and production storage estimates.

### Gate 5: production and audit

- run the chosen trajectory count with resumable output;
- validate every raw trajectory before HDF5 conversion;
- validate HDF5 checksums, metadata, splits, statistics, and random sample plots;
- retain resolved configs, package versions, commit, logs, checksums, and QA summary;
- mark a dataset immutable only after the audit passes.

## Implementation sequence

1. Add common configuration/backend/spectral utilities and preserve notebook equations exactly.
2. Implement the four per-case generators with NumPy first and smoke-test them against notebook outputs.
3. Add the Torch backend, explicit GPU selection, trajectory batching, CPU streaming, and CPU/GPU tests.
4. Add atomic raw output, manifests, resume semantics, checksums, storage estimation, and safe symlink setup.
5. Implement the unified HDF5 postprocessor and compatibility/schema validator.
6. Implement visualization/QA scripts and reproduce each notebook's diagnostic panels.
7. Run numerical qualification at 192 x 192 and select precision/time-step defaults from evidence.
8. Run a pilot dataset, revise storage/batch defaults, then launch production.

## Completion criteria for the next implementation phase

The generation system is ready for a formal run only when:

- all four scripts can launch on a chosen CPU or assigned GPU with documented parameters;
- the two 2D outputs are native 192 x 192 simulations;
- raw runs are atomic, resumable, reproducible, and stored through the `/data/wanglz` link;
- one postprocessor produces schema-valid, loader-compatible HDF5 for every case;
- splits are trajectory-based and statistics are training-only;
- notebook-equivalent visual and physical QA reports are reproducible from saved data;
- CPU/GPU, time-step, and 192/256 validation gates pass;
- a pilot run confirms runtime, memory, and storage estimates before production scale is chosen.

## Repository note

The current root `.gitignore` ignores the entire `datagen/` directory, and `git ls-files datagen` is empty. Before these examples become canonical, explicitly decide which code, plans, tests, and lightweight configs under `datagen/` should be version-controlled while continuing to ignore `datagen/data` and generated artifacts.

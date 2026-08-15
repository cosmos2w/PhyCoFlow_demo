# PhyCoFlow canonical dataset scripts

These scripts turn the six demonstration notebooks into reproducible raw
trajectories, unified HDF5 tensor datasets, and physical-coherence QA figures.
They do not generate data merely by being imported.

## Environment and storage

Use `phycoflow_env`. It contains NumPy/SciPy, CUDA PyTorch, h5py, Matplotlib,
PyYAML, tqdm, and pytest. The four spectral solvers support assigned CUDA GPUs;
the electro-thermal and Elder-type solvers use SciPy sparse matrices and run on
CPU with `--backend numpy --device cpu`.

To check or create the optional large-data link:

```bash
conda run -n phycoflow_env python datagen/setup_data_root.py --check-only
conda run -n phycoflow_env python datagen/setup_data_root.py
```

The default link is `datagen/data -> /data/wanglz/PhyCoFlow/datagen`. Every
generation command prints an estimated raw-data size and available disk space
before it starts.

## Three-step workflow

Each case directory contains independent `generate.py`, post-processing, and
`visualize.py` entry points. The two multiphysics cases use the explicitly
requested filename `postprocessing.py`; the first four use `postprocess.py`.
Every file begins with case-specific commands and supports `--help`.

### 1. Generate raw trajectories on an assigned GPU

Assign physical GPU 1 with `CUDA_VISIBLE_DEVICES=1`; inside the process it is
logical `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n phycoflow_env \
  python datagen/1_burgers/generate.py --device cuda:0 --num-trajectories 100

CUDA_VISIBLE_DEVICES=1 conda run -n phycoflow_env \
  python datagen/2_ks/generate.py --device cuda:0 --num-trajectories 100

CUDA_VISIBLE_DEVICES=1 conda run -n phycoflow_env \
  python datagen/3_brusselator/generate.py --device cuda:0 \
  --resolution 192 --num-trajectories 100

CUDA_VISIBLE_DEVICES=1 conda run -n phycoflow_env \
  python datagen/4_navier_stokes/generate.py --device cuda:0 \
  --resolution 192 --num-trajectories 100

conda run -n phycoflow_env \
  python datagen/5_electro_thermal/generate.py --backend numpy --device cpu \
    --resolution 128 --num-trajectories 16 --workers 4

conda run -n phycoflow_env \
  python datagen/6_mass_transport_fluid/generate.py --backend numpy --device cpu \
  --resolution 128 --record-time 20 --dt 0.25 --save-every 8 \
  --num-trajectories 8
```

There are two nested progress displays: overall completed trajectories and the
current trajectory's numerical steps. `--resume` skips only checksum-validated
trajectories. Use `--dry-run` to inspect the complete resolved configuration and
storage estimate without creating a directory.

Electro-thermal generation accepts `--workers N` to solve independent steady
realizations in separate CPU processes. The parent process alone writes the
checksummed trajectory files and manifest, so interrupted runs remain safely
resumable. Start with `--workers 4` at resolution 128 and increase only if RAM
allows; the worker count may be changed when using `--resume` without changing
the physical dataset.

Important tunable options include trajectory count/seeds, resolution, `dt`,
burn-in/record time, save cadence, solver/storage precision, domain length, and
all case-specific physical parameters. The original periodic 2D defaults are
192 x 192; both nonperiodic multiphysics defaults are boundary-inclusive
128 x 128 grids. NumPy CPU reference runs use `--backend numpy --device cpu`.
Multiphysics conditions are indexed points from deterministic scrambled Sobol
designs, so trajectory IDs retain comparable physical-parameter coverage.

Raw layout:

```text
<raw-dir>/
  manifest.json
  resolved_config.json
  trajectories/
    trajectory_000000.npz   # state, time, step, x, optional y
    trajectory_000000.json  # seed, physics, provenance, checksum, diagnostics
```

Kolmogorov raw files store the solver-native vorticity. Velocity and zero-mean
pressure are derived during postprocessing. Electro-thermal raw files also
store the material mask, conductivity, Joule heating, and thermal conductivity.
Mass-transport raw files retain source, pressure, diagnostic face velocities,
nonlinear iteration counts, and adaptive-step counts.

### 2. Convert raw trajectories to unified HDF5

```bash
conda run -n phycoflow_env python datagen/1_burgers/postprocess.py \
  --raw-dir datagen/data/raw/burgers/burgers_canonical \
  --output datagen/data/processed/burgers.h5
```

Use the equivalent script in another case directory. The processor shows its
current trajectory, validates grids/checksums/finite values/splits, prints a
summary, and writes `<output-stem>_README.md` beside the HDF5 file.

Unified core structure:

```text
fields       [B,T,N,1,1,C] float32
coordinates  [N,1,1,3]     float32
time         [T]            float64
conditions   [B,P]          float32
trajectory_id, seed, splits/{train,validation,test}
statistics/{train_mean,train_std}
statistics/{channel_offset,channel_scale_99}  # multiphysics cases
diagnostics/<physical metric>
auxiliary/vorticity         # Kolmogorov flow unless --no-auxiliary
auxiliary/{ellipse_mask,conductivity,joule_heating,thermal_conductivity}
auxiliary/{source_field,pressure}
metadata/json and root provenance attributes
```

Splits are trajectory-based. Statistics use only training trajectories. Fields
and coordinates remain in physical units.

### 3. Extract and visualize saved fields

```bash
conda run -n phycoflow_env python datagen/3_brusselator/visualize.py \
  --input datagen/data/processed/brusselator.h5 \
  --trajectory 0 --time-index -1 --output brusselator_qa.png
```

`--input` may also be one raw trajectory NPZ. Output can be PNG, PDF, or SVG.
Each six-panel figure reproduces the notebook's key field, distribution,
spectrum, joint-field/topology, and PDE/constraint evidence; the command also
prints numerical diagnostics.

Case QA includes:

- Burgers: space-time field, shock gradient, spectrum, mean drift, PDE residual.
- KS: space-time chaos, spectrum, phase portrait, variance, PDE residual.
- Brusselator: `u/v`, joint cloud, marginal/radial spectra, level sets,
  positivity and two PDE residuals.
- Kolmogorov flow: `u/v/p/omega`, joint cloud, kinetic-energy spectrum,
  divergence, pressure gauge, momentum residuals, energy, and enstrophy.
- Electro-thermal: complex `E_z`, temperature rise, ellipse geometry, Joule
  heating, coupled spectra, Helmholtz/heat residuals, and Robin mismatch.
- Mass transport-fluid: concentration, `u_x/u_y`, source-relative streamlines,
  accumulation/maximum-speed histories, flow/transport spectra, conservative
  residuals, boundary flux, pressure gauge, and nonlinear step counts.

## Validation tests

Run the lightweight suite without producing a formal dataset:

```bash
conda run -n phycoflow_env pytest -q datagen/tests
```

The tests use temporary directories and cover all six CPU solvers, raw
checksums, HDF5 schemas, and PNG rendering. A separate CUDA smoke command is
listed in the test file for the four spectral solvers and can be selected with
`CUDA_VISIBLE_DEVICES`.

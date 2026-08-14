# Canonical compact multiphysics cases: implementation specification

## Purpose

This is the implementation contract for adding:

1. `electro_thermal/viz.ipynb`
2. `mass_transport_fluid/viz.ipynb`

The cases must be based on the models in *Multiphysics Bench: Benchmarking and Investigating Scientific Machine Learning for Multiphysics PDEs* and its released code, but adapted into small, deterministic, self-contained reference datasets for multi-field sparse reconstruction.

Do not reproduce or download the full Multiphysics Bench dataset. The full release is about 31.8 GB and was generated with licensed COMSOL Multiphysics 6.2 plus MATLAB LiveLink. These notebooks must run without COMSOL and must not silently download external data.

## Authoritative references and version pin

- Paper: [arXiv:2505.17575v1](https://arxiv.org/abs/2505.17575v1)
- Code: [xie-lab-ml/multiphysics-bench](https://github.com/xie-lab-ml/multiphysics-bench)
- Code revision: [`1195b44915712929d605ee5076a831c7124ead93`](https://github.com/xie-lab-ml/multiphysics-bench/commit/1195b44915712929d605ee5076a831c7124ead93)
- License: MIT; retain attribution in both notebooks.

Relevant upstream files:

- [`TE_heat.m`](https://github.com/xie-lab-ml/multiphysics-bench/blob/1195b44915712929d605ee5076a831c7124ead93/DataProcessing/data_generate/TE_heat/TE_heat.m)
- [`main_TE_heat.m`](https://github.com/xie-lab-ml/multiphysics-bench/blob/1195b44915712929d605ee5076a831c7124ead93/DataProcessing/data_generate/TE_heat/main_TE_heat.m)
- [`parm2matrix_TE_heat.m`](https://github.com/xie-lab-ml/multiphysics-bench/blob/1195b44915712929d605ee5076a831c7124ead93/DataProcessing/data_generate/TE_heat/parm2matrix_TE_heat.m)
- [`merge_data_TE_heat.py`](https://github.com/xie-lab-ml/multiphysics-bench/blob/1195b44915712929d605ee5076a831c7124ead93/DataProcessing/Preprocess/merge_data_TE_heat.py)
- [`generate_TE_heat.py`](https://github.com/xie-lab-ml/multiphysics-bench/blob/1195b44915712929d605ee5076a831c7124ead93/DiffusionPDE/scripts/generate_TE_heat.py)
- [`Elder.m`](https://github.com/xie-lab-ml/multiphysics-bench/blob/1195b44915712929d605ee5076a831c7124ead93/DataProcessing/data_generate/Elder/Elder.m)
- [`main_Elder.m`](https://github.com/xie-lab-ml/multiphysics-bench/blob/1195b44915712929d605ee5076a831c7124ead93/DataProcessing/data_generate/Elder/main_Elder.m)
- [`parm2matrix_Elder.m`](https://github.com/xie-lab-ml/multiphysics-bench/blob/1195b44915712929d605ee5076a831c7124ead93/DataProcessing/data_generate/Elder/parm2matrix_Elder.m)
- [`merge_data_Elder.py`](https://github.com/xie-lab-ml/multiphysics-bench/blob/1195b44915712929d605ee5076a831c7124ead93/DataProcessing/Preprocess/merge_data_Elder.py)
- [`generate_Elder.py`](https://github.com/xie-lab-ml/multiphysics-bench/blob/1195b44915712929d605ee5076a831c7124ead93/DiffusionPDE/scripts/generate_Elder.py)

When sources disagree, use this precedence:

1. Governing model and coupling direction from the paper.
2. Executable parameter assignments and geometry in the pinned generation code.
3. Channel definitions and sparse-observation behavior in the pinned preprocessing/DiffusionPDE code.
4. Comments and prose only when consistent with the above.

State every numerical adaptation from COMSOL/FEM in the notebook. Do not claim bitwise or FEM equivalence. Call each result a **compact open-source canonical adaptation of the Multiphysics Bench model**.

## Required repository changes

Create:

```text
4_electro_thermal/
└── viz.ipynb
5_mass_transport_fluid/
└── viz.ipynb
```

Update the table and conventions in the root `viz.ipynb`:

- Add both cases to the active suite.
- Describe electro-thermal as a steady ensemble and mass transport-fluid as a transient ensemble.
- Extend the array convention:
  - one transient realization: `[time, channel, y, x]`
  - steady ensemble: `[sample, channel, y, x]`
  - transient ensemble: `[sample, time, channel, y, x]`
- Keep the warning that topology plots are previews, not persistent-homology implementations.

Do not add the upstream repository, trained models, the full official dataset, COMSOL files, or generated solver matrices.

## Shared notebook rules

Both notebooks must follow the existing structure:

1. Markdown introduction with PDEs, domain, fields, coupling, provenance, and adaptation note.
2. Imports and numerical parameters near the top.
3. Deterministic field generation.
4. Sparse-observation construction.
5. Numerical/data diagnostics returned as a `diagnostics` dictionary.
6. A Matplotlib `FuncAnimation` dashboard shown only outside smoke mode.

Non-negotiable conventions:

- Use only Python, NumPy, SciPy, Matplotlib, and IPython. No PyTorch, COMSOL, MATLAB, FEniCS, internet, or GPU dependency.
- Use `PHYCOFLOW_VIZ_SMOKE=1` for reduced execution.
- Solve in float64/complex128; optional export uses float32.
- Use fixed seeds and deterministic quasi-random parameter designs.
- Grids include physical boundaries for these non-periodic cases.
- Keep physical units in `fields`; normalization is a separate view and metadata object.
- Never compute normalization independently for each sample.
- Save nothing by default. An explicit `save_compact_dataset = False` flag may enable export.
- Clear notebook outputs before delivery; do not commit embedded JavaScript animations.
- Do not hide failures with blanket clipping. Tiny negative concentration at solver tolerance may be cleaned only after recording its magnitude.
- No output file may exceed 64 MiB; combined default target fields should be below 32 MiB before compression.

Use:

```python
electro_thermal_seed = 23
mass_transport_seed = 29
```

Use `scipy.stats.qmc.Sobol(scramble=True, seed=seed)` for parameter coverage. Normal sample counts are powers of two.

## Sparse-reconstruction protocol

The primary task is **full multi-field reconstruction from sparse spatial measurements**, not operator learning or future-state forecasting.

### Primary sensor budget

Use exactly 500 distinct grid points per field on a 128 x 128 grid (`500 / 16384 = 0.030517578125`). This follows the released DiffusionPDE scripts. Generate indices with `numpy.random.default_rng(seed).choice(H * W, 500, replace=False)`; do not mutate NumPy's global RNG.

Provide two protocols:

1. `official_independent_500` (primary)
   - Electro-thermal: one mask for both real and imaginary `E_z` using seed 1; a separate `T` mask using seed 0. The electric channels are co-located because they form one complex observation.
   - Mass transport-fluid: `u_x` seed 2, `u_y` seed 1, `c` seed 0. Reuse each channel's mask at every time.
   - Optional unknown-conditioning experiment: electro-thermal material seed 2; mass source seed 3.
2. `shared_500` (required ablation)
   - Seed 0 and the same 500 locations for every target channel.

Masks are Boolean arrays separate from values. Never encode unobserved values as physical zero without a mask.

Expose at minimum:

```python
sensor_masks       # [protocol, channel, y, x]
observations       # np.where(mask, normalized_fields, np.nan)
sensor_coordinates # exact physical coordinates
```

For the transient case, broadcast masks over sample/time; do not duplicate them in an archive.

### Reconstruction metrics

Supply reusable metric functions, but no reconstruction model:

- relative L2 and normalized RMSE per channel on **unobserved points only**
- observed-point consistency error
- joint sliced-Wasserstein input arrays shaped `[point, channel]`
- radial spectral-error inputs
- PDE residual diagnostics

Average channels only after scaling, and also report every channel. Default observations are noiseless; optional noise must be explicit and seeded.

## Case 1: electro-thermal coupling

### Model

Use the paper's two-dimensional TE-polarized scalar reduction:

$$
\nabla^2 E_z+k_0^2\mu_r\left(\varepsilon_r-j\frac{\sigma(T)}{\omega\varepsilon_0}\right)E_z=0,
$$

$$
\nabla\cdot(\kappa\nabla T)+\frac12\sigma(T)|E_z|^2=0.
$$

The electromagnetic sign matches the pinned Python physics-loss code; document the assumed harmonic time convention.

Inside silicon:

$$
\sigma(T)=1.602\,\Sigma_{\mathrm{Si}}\exp\left[-\frac{1.12}{(8.6173\times10^{-5})T}\right]\ \mathrm{S/m}.
$$

Outside, use alumina and electrical conductivity $10^{-7}$ S/m. Coupling is bidirectional: `E_z` produces Joule heating, and `T` changes silicon conductivity.

### Geometry and constants

Use SI units internally.

| Quantity | Canonical value |
|---|---:|
| Interior alumina square | 0.128 m x 0.128 m |
| Square including absorbing layer | 0.148 m x 0.148 m |
| Absorbing-layer thickness | 0.010 m |
| Normal output resolution | 128 x 128 after cropping |
| Silicon semi-major axis `a` | [0.020, 0.030] m |
| Silicon semi-minor axis `b` | [0.010, 0.020] m |
| Ellipse angle `phi` | [0, 2*pi) |
| Silicon prefactor `Sigma_Si` | [1e11, 3e11] S/m |
| Alumina thermal conductivity | [10, 20] W/(m K) |
| Silicon thermal conductivity | 70 W/(m K) |
| Relative permittivity, silicon/alumina | 11.7 / 1.0 |
| Relative permeability | 1.0 |
| Frequency | 4e9 Hz |
| Incident amplitude | 3e5 V/m |
| Incident angle | pi/3 |
| Ambient temperature | 293.15 K |
| Convective coefficient | 15 W/(m^2 K) |

Use 16 Sobol samples in normal mode and 2 samples on a reduced grid in smoke mode.

### Numerical adaptation

Implement with SciPy sparse matrices:

1. Build the cell-centered rotated-ellipse mask as upstream does.
2. Represent the incident TE wave explicitly.
3. Solve the complex scalar Helmholtz equation with a second-order stencil and complex-coordinate-stretch absorbing layer on the 0.148 m square. Crop the absorbing layer.
4. Compute `q_J = 0.5 * sigma(T) * abs(E_z)**2`.
5. Solve variable-conductivity heat diffusion on the physical square with `-kappa * dT/dn = h * (T - T_ext)`.
6. Picard-iterate the electric and thermal solves with under-relaxation until relative infinity-norm changes in both `T` and `sigma` are below `1e-6`; fail loudly after 30 iterations.

The absorbing layer is the largest COMSOL-to-open-source departure. Document its formula and parameters. Zero-Dirichlet electric boundaries are not acceptable because they change the scattering problem.

### Arrays

```python
fields.shape == (16, 3, 128, 128)
channel_names == ("E_z_real", "E_z_imag", "temperature")
channel_units == ("V/m", "V/m", "K")
sample_parameters.shape == (16, 5)
parameter_names == ("a", "b", "phi", "Sigma_Si", "kappa_alumina")
ellipse_masks.shape == (16, 128, 128)
```

The upstream `mater` channel mixes silicon conductivity prefactor inside the ellipse with alumina thermal conductivity outside. Recreate it only as `legacy_material_map`; the canonical conditioning is the ellipse mask plus scalar parameters.

### Diagnostics

Include:

- `finite`
- `maximum_coupling_iterations`
- `maximum_relative_temperature_update`
- `maximum_relative_conductivity_update`
- `relative_helmholtz_residual` on the physical interior, excluding two boundary cells
- `relative_heat_residual` on the same interior
- `relative_robin_boundary_residual`
- `minimum_temperature`, `maximum_temperature`
- `minimum_abs_electric_field`, `maximum_abs_electric_field`
- `sensor_count_per_channel`
- `deterministic_parameter_hash`

Require finite values, converged coupling, 500 unique sensors per target field, and no temperature below ambient beyond tolerance. Normalize residuals by the sum of constituent-term norms, not one term.

### Dashboard

Animate/slide across at most 16 samples in a 2 x 3 layout:

1. `Re(E_z)` with sensor points.
2. `Im(E_z)` with the same sensor points.
3. `T` with temperature sensors.
4. Joint `(Re(E_z), Im(E_z), T)` cloud.
5. Shell-averaged electric and temperature spectra, labeled accurately.
6. Ellipse boundary plus temperature and `|E_z|` level sets.

## Case 2: mass transport-fluid coupling (Elder type)

### Model

Use:

$$
\epsilon\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0,
$$

$$
\mathbf{u}=-\frac{K}{\mu}(\nabla p-\rho\mathbf{g}),\qquad
\rho(c)=\rho_0+\beta\max(c,0),
$$

$$
\epsilon\frac{\partial c}{\partial t}+\mathbf{u}\cdot\nabla c-
\nabla\cdot(\epsilon D_L\nabla c)=S_c.
$$

This combines the paper's balance/transport equations with the Darcy relation configured by the COMSOL model. `max(c,0)` matches the released density expression; it must not hide a materially unstable solve.

The source is:

$$
S_c(x,y)=\frac{A}{365\cdot24\cdot60\cdot60}
\exp\left[-\frac{(x-x_0)^2+(y-y_0)^2}{2s^2}\right].
$$

### Geometry, constants, and parameters

Use centered COMSOL coordinates `x in [-150,150] m`, `y in [-75,75] m`. The paper's `[0,300] x [0,150]` rectangle is the translated equivalent.

| Quantity | Canonical value |
|---|---:|
| `L` | 150 m |
| Normal output resolution | 128 x 128 |
| Output times | 0, 2, ..., 20 years (11 frames) |
| Pure-water density `rho0` | 1000 kg/m^3 |
| Brine density at `c=1` | 1200 kg/m^3 |
| Density coefficient `beta` | 200 kg/mol |
| Dynamic viscosity `mu` | 1e-3 Pa s |
| Permeability `K` | 500 mD = 4.9346165e-13 m^2 |
| Porosity/fluid fraction `epsilon` | 0.1 |
| Effective molecular diffusion `D_L` | 3.56e-6 m^2/s |
| Top-right concentration `c_s` | 1 mol/m^3 |
| Initial concentration | 0 mol/m^3 |
| Source amplitude `A` | [1e-3, 8e-3] |
| Source center `x0` | [-70, 70] m |
| Source center `y0` | [-30, 30] m |
| Source width `s` | [10, 70] m |

Use 8 Sobol source samples in normal mode. Smoke mode uses one reduced-grid trajectory ending at year 4 with at least three frames.

### Boundary and gauge conditions

Concentration:

- bottom: `c = 0`
- top-right half (`x >= 0`): `c = 1`
- top-left, left, and right: zero normal total-species flux/symmetry

Darcy flow:

- zero normal mass flux on the rectangle
- zero spatial-mean pressure gauge after every solve
- hydrostatic initial pressure and consistent initial velocity

### Numerical adaptation

Use conservative cell-centered finite volume or equivalent flux form. Unstabilized pointwise central advection is not acceptable.

Required algorithm:

1. Face-centered Darcy mass fluxes; cell-centered `p`, `c`, and `rho`.
2. Picard-couple pressure/density and concentration inside every step.
3. Upwind or monotone-limited advection plus implicit diffusion.
4. Adaptive internal steps from advective CFL and nonlinear convergence; output only at the 11 canonical times.
5. Boundary conditions through numerical fluxes.
6. Record iteration counts and reduced/rejected time steps.

Exact FEM values are not expected. Preserve equations, domain, parameters, boundaries, coupling, cadence, and channel semantics.

### Arrays

```python
trajectories.shape == (8, 11, 3, 128, 128)
fields = trajectories[0]
fields.shape == (11, 3, 128, 128)
channel_names == ("u_x", "u_y", "concentration")
channel_units == ("m/s", "m/s", "mol/m^3")
times_years == np.arange(0.0, 22.0, 2.0)
source_fields.shape == (8, 128, 128)
sample_parameters.shape == (8, 4)
parameter_names == ("A", "x0", "y0", "s")
```

Do not flatten 11 times into 33 channels internally. Upstream merged order was source, all `u_x`, all `u_y`, then all `c`; provide only an optional compatibility converter.

### Diagnostics

Include:

- `finite`
- `minimum_concentration_before_tolerance_cleanup`, `maximum_concentration`
- `maximum_density`, `maximum_speed`
- `relative_mass_balance_residual`
- `relative_transport_residual`
- `relative_boundary_flux_residual`
- `maximum_pressure_mean`
- `maximum_picard_iterations`
- `reduced_timestep_count`
- `sensor_count_per_channel`
- `deterministic_parameter_hash`

Use solver fluxes for residuals. Exclude first/last recorded time for centered temporal diagnostics. Normalize by the sum of constituent-term norms and evaluate boundaries separately.

### Dashboard

Animate the first trajectory over 11 times in a 2 x 3 layout:

1. Concentration with sensors.
2. `u_x` with sensors.
3. `u_y` with sensors.
4. Joint `(u_x, u_y, c)` cloud.
5. Kinetic-energy and concentration shell-averaged spectra.
6. Concentration level sets, velocity streamlines, and source center.

Expose `trajectory_index` near the top for inspecting another realization.

## Normalization and compact export

Keep raw physical values in `fields`/`trajectories`. Provide:

```python
normalized = (raw - channel_offset) / channel_scale
```

Rules:

- One fixed offset/scale per channel for the whole ensemble, never per sample/frame.
- Estimate robust scales from only the first half of the deterministic ensemble using the 99th percentile of absolute centered values; store constants.
- Center temperature at 293.15 K.
- Concentration may use fixed scale `c_s = 1`.
- Correct affine inverse is `(z + 0.9) / 1.8 * (max - min) + min`.
- Never use a full evaluation target to normalize itself independently.

If `save_compact_dataset` is enabled, write one compressed `.npz` and one `.json` per case. Do not duplicate normalized fields.

Required numeric keys:

```text
fields or trajectories
x
y
times                     # transient only
sample_parameters
source_fields             # mass transport only
ellipse_masks             # electro-thermal only
sensor_indices_official
sensor_indices_shared
channel_offset
channel_scale
```

JSON must include names, units, seeds, version label, source URLs, pinned commit, grids, solver tolerances, and SHA-256 of the NPZ.

Expected raw target storage:

- Electro-thermal: about 3.0 MiB (`16*3*128*128*4`).
- Mass transport-fluid: about 16.5 MiB (`8*11*3*128*128*4`).
- Combined: about 19.5 MiB before compression.

## Upstream discrepancies that must be documented

1. Paper ellipse axes are labeled meters; model geometry is millimeters. Use 20-30 mm and 10-20 mm.
2. Paper/code conductivity ranges use different factorizations. Use executable `1.602*Sigma_Si*exp(...)`, `Sigma_Si in [1e11,3e11]`.
3. TE scripts contain run literals for 1, 3,000, and 10,000 samples. They are not physical requirements.
4. MATLAB called/file/function names have suffix mismatches (`TE_heat`/`TE_heat_v4`, `Elder`/`Elder_v1`). Do not reproduce them.
5. Paper main table says 10,000/1,000 for all cases, while appendix/repository use 1,000/100 for Elder. This adaptation intentionally uses eight trajectories.
6. Elder source-width comment says `[10,50]`; executable `10+60*rand` gives `[10,70]`. Use `[10,70]`.
7. Paper Elder table is normalized/abbreviated; use executable physical center/width ranges above.
8. Released Elder inverse normalization omits `+min_val`. Use the correct inverse.
9. Released physics-loss code uses zero-padded derivatives that contaminate boundaries. Separate interior PDE and boundary residuals.
10. Upstream comments call `delta_x` 1 m while executable value is `300/128`; rectangular `dx` and `dy` differ. Use coordinates, not comments.

## Acceptance checklist

- [ ] Both notebooks run top-to-bottom in smoke mode without downloads/proprietary software.
- [ ] Normal mode has the stated 128 x 128 grids and deterministic sample counts.
- [ ] Repeated smoke runs match within solver tolerance and parameter hashes match exactly.
- [ ] Required shapes, axes, names, units, and finite checks pass.
- [ ] Electro-thermal coupling is iterated bidirectionally.
- [ ] Mass velocity updates from concentration-dependent density.
- [ ] Every target field has exactly 500 unique primary sensors.
- [ ] `E_z` real/imaginary sensor locations match.
- [ ] Transient spatial masks remain fixed through time.
- [ ] Metrics separate observed and unobserved points.
- [ ] PDE and boundary residuals are normalized and separate.
- [ ] No full dataset, checkpoint, or COMSOL artifact is downloaded/committed.
- [ ] Notebook outputs are cleared.
- [ ] Optional archives satisfy size, checksum, and provenance requirements.
- [ ] Root `viz.ipynb` is updated without removing existing cases.

## Implementation order for the next Codex agent

1. Read every existing notebook and this document before editing.
2. Implement deterministic mask/metric helpers explicitly in each notebook so both remain independently runnable.
3. Implement electro-thermal smoke mode, then normal mode and diagnostics.
4. Implement mass transport smoke mode, then normal mode and diagnostics.
5. Add dashboards only after numerical diagnostics pass.
6. Add optional export and verify archive shapes/checksums.
7. Update the root index last.
8. Execute both notebooks in smoke mode from clean kernels, inspect diagnostics, clear outputs, and report any untested normal-mode runtime.

Do not weaken coupled physics merely to make a residual small. If the specified discretization is unstable or too slow, preserve the equations and data contract, document the blocker, and improve the solver rather than silently changing the benchmark.

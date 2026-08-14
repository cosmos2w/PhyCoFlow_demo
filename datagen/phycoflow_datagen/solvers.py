"""Spectral numerical solvers derived from the four demonstration notebooks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import qmc

from .backend import ArrayBackend


ProgressFactory = Callable[..., Iterable[int]]


def _steps(duration: float, dt: float, label: str) -> int:
    count = int(round(duration / dt))
    if count < 0 or not np.isclose(count * dt, duration, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{label}={duration} must be a nonnegative integer multiple of dt={dt}")
    return count


def _iter_steps(total: int, progress: ProgressFactory | None, description: str):
    values = range(1, total + 1)
    return values if progress is None else progress(values, total=total, desc=description)


def _etdrk4_coefficients(ops: ArrayBackend, linear, dt: float, contour_points: int = 16):
    roots = ops.exp(
        1j * np.pi * (ops.arange(1, contour_points + 1) - 0.5) / contour_points
    )
    lr = dt * linear[..., None] + roots
    e = ops.exp(dt * linear)
    e2 = ops.exp(0.5 * dt * linear)
    q = dt * ops.mean((ops.exp(lr / 2.0) - 1.0) / lr, axis=-1).real
    f1 = dt * ops.mean(
        (-4.0 - lr + ops.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3,
        axis=-1,
    ).real
    f2 = dt * ops.mean(
        (2.0 + lr + ops.exp(lr) * (-2.0 + lr)) / lr**3,
        axis=-1,
    ).real
    f3 = dt * ops.mean(
        (-4.0 - 3.0 * lr - lr**2 + ops.exp(lr) * (4.0 - lr)) / lr**3,
        axis=-1,
    ).real
    return e, e2, q, f1, f2, f3


def _etdrk4_step(state_hat, nonlinear, coefficients):
    e, e2, q, f1, f2, f3 = coefficients
    nv = nonlinear(state_hat)
    a = e2 * state_hat + q * nv
    na = nonlinear(a)
    b = e2 * state_hat + q * na
    nb = nonlinear(b)
    c = e2 * a + q * (2.0 * nb - nv)
    nc = nonlinear(c)
    return e * state_hat + f1 * nv + 2.0 * f2 * (na + nb) + f3 * nc


def _append_snapshot(snapshots: list[np.ndarray], value, ops: ArrayBackend, dtype: np.dtype) -> None:
    snapshots.append(ops.to_numpy(value).astype(dtype, copy=False))


def solve_burgers(
    config: dict[str, Any], seed: int, ops: ArrayBackend, progress: ProgressFactory | None
) -> dict[str, np.ndarray]:
    n = config["resolution"]
    length = config["domain_length"]
    dt = config["dt"]
    viscosity = config["viscosity"]
    burn_steps = _steps(config["burn_in_time"], dt, "burn_in_time")
    record_steps = _steps(config["record_time"], dt, "record_time")
    save_every = config["save_every"]
    storage_dtype = np.dtype(config["storage_dtype"])

    x_np = np.linspace(0.0, length, n, endpoint=False, dtype=np.float64)
    rng = np.random.default_rng(seed)
    amplitudes = np.array([-1.0, 0.25, 0.10], dtype=np.float64)
    phases = np.array([0.0, 0.3, -0.2], dtype=np.float64)
    amplitudes *= 1.0 + config["ic_amplitude_jitter"] * rng.standard_normal(3)
    phases += config["ic_phase_jitter"] * rng.standard_normal(3)
    u0_np = sum(
        amplitudes[index] * np.sin((index + 1) * x_np + phases[index])
        for index in range(3)
    )

    dx = length / n
    wave = 2.0 * np.pi * ops.fftfreq(n, d=dx)
    modes = ops.fftfreq(n, d=1.0 / n)
    dealias = abs(modes) <= n / 3
    linear = -viscosity * wave**2

    def nonlinear(u_hat):
        u = ops.ifft(u_hat).real
        return -0.5j * wave * ops.fft(u**2) * dealias

    coefficients = _etdrk4_coefficients(ops, linear, dt, config["contour_points"])
    u_hat = ops.fft(ops.asarray(u0_np)) * dealias
    snapshots: list[np.ndarray] = []
    times: list[float] = []
    saved_steps: list[int] = []
    if burn_steps == 0:
        _append_snapshot(snapshots, ops.ifft(u_hat).real[None, :], ops, storage_dtype)
        times.append(0.0)
        saved_steps.append(0)

    total_steps = burn_steps + record_steps
    for step in _iter_steps(total_steps, progress, "Burgers burn-in + solver steps"):
        u_hat = _etdrk4_step(u_hat, nonlinear, coefficients) * dealias
        relative_step = step - burn_steps
        if step >= burn_steps and (
            relative_step % save_every == 0 or relative_step == record_steps
        ):
            _append_snapshot(snapshots, ops.ifft(u_hat).real[None, :], ops, storage_dtype)
            times.append(relative_step * dt)
            saved_steps.append(relative_step)

    ops.synchronize()
    return {
        "state": np.stack(snapshots),
        "time": np.asarray(times, dtype=np.float64),
        "step": np.asarray(saved_steps, dtype=np.int64),
        "x": x_np,
    }


def solve_ks(
    config: dict[str, Any], seed: int, ops: ArrayBackend, progress: ProgressFactory | None
) -> dict[str, np.ndarray]:
    n = config["resolution"]
    length = config["domain_length"]
    dt = config["dt"]
    burn_steps = _steps(config["burn_in_time"], dt, "burn_in_time")
    record_steps = _steps(config["record_time"], dt, "record_time")
    save_every = config["save_every"]
    storage_dtype = np.dtype(config["storage_dtype"])

    x_np = np.linspace(0.0, length, n, endpoint=False, dtype=np.float64)
    dx = length / n
    wave_np = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    modes_np = np.fft.fftfreq(n) * n
    dealias_np = np.abs(modes_np) <= n / 3
    rng = np.random.default_rng(seed)
    base = np.cos(2.0 * np.pi * x_np / length) * (1.0 + np.sin(2.0 * np.pi * x_np / length))
    noise_hat = np.fft.fft(rng.standard_normal(n))
    noise_hat *= np.exp(-config["noise_filter"] * wave_np**4) * dealias_np
    perturbation = np.fft.ifft(noise_hat).real
    perturbation -= perturbation.mean()
    u0_np = base + config["noise_amplitude"] * perturbation / perturbation.std()

    wave = ops.asarray(wave_np)
    dealias = ops.asarray(dealias_np.astype(np.float64)) > 0.5
    linear = (
        config["second_order_coefficient"] * wave**2
        - config["fourth_order_coefficient"] * wave**4
    )

    def nonlinear(u_hat):
        u = ops.ifft(u_hat).real
        return -0.5j * config["advection_coefficient"] * wave * ops.fft(u**2) * dealias

    coefficients = _etdrk4_coefficients(ops, linear, dt, config["contour_points"])
    u_hat = ops.fft(ops.asarray(u0_np)) * dealias
    snapshots: list[np.ndarray] = []
    times: list[float] = []
    saved_steps: list[int] = []
    total_steps = burn_steps + record_steps
    if burn_steps == 0:
        _append_snapshot(snapshots, ops.ifft(u_hat).real[None, :], ops, storage_dtype)
        times.append(0.0)
        saved_steps.append(0)

    for step in _iter_steps(total_steps, progress, "KS burn-in + solver steps"):
        u_hat = _etdrk4_step(u_hat, nonlinear, coefficients) * dealias
        relative_step = step - burn_steps
        if step >= burn_steps and (
            relative_step % save_every == 0 or relative_step == record_steps
        ):
            _append_snapshot(snapshots, ops.ifft(u_hat).real[None, :], ops, storage_dtype)
            times.append(relative_step * dt)
            saved_steps.append(relative_step)

    ops.synchronize()
    return {
        "state": np.stack(snapshots),
        "time": np.asarray(times, dtype=np.float64),
        "step": np.asarray(saved_steps, dtype=np.int64),
        "x": x_np,
    }


def solve_brusselator(
    config: dict[str, Any], seed: int, ops: ArrayBackend, progress: ProgressFactory | None
) -> dict[str, np.ndarray]:
    n = config["resolution"]
    length = config["domain_length"]
    dt = config["dt"]
    burn_steps = _steps(config["burn_in_time"], dt, "burn_in_time")
    record_steps = _steps(config["record_time"], dt, "record_time")
    save_every = config["save_every"]
    storage_dtype = np.dtype(config["storage_dtype"])
    a_value, b_value = config["A"], config["B"]

    x_np = np.linspace(0.0, length, n, endpoint=False, dtype=np.float64)
    y_np = np.linspace(0.0, length, n, endpoint=False, dtype=np.float64)
    dx = length / n
    wave_np = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx_np, ky_np = np.meshgrid(wave_np, wave_np, indexing="xy")
    k2_np = kx_np**2 + ky_np**2
    rng = np.random.default_rng(seed)

    def smooth_noise() -> np.ndarray:
        raw_hat = np.fft.fft2(rng.standard_normal((n, n)))
        field = np.fft.ifft2(raw_hat * np.exp(-config["noise_filter"] * k2_np**2)).real
        field -= field.mean()
        return config["noise_amplitude"] * field / field.std()

    u = ops.asarray(a_value + smooth_noise())
    v = ops.asarray(b_value / a_value + smooth_noise())
    k2 = ops.asarray(k2_np)
    diffuse_u = ops.exp(-0.5 * dt * config["diffusivity_u"] * k2)
    diffuse_v = ops.exp(-0.5 * dt * config["diffusivity_v"] * k2)

    def reaction(u_value, v_value):
        coupling = u_value**2 * v_value
        return (
            a_value - (b_value + 1.0) * u_value + coupling,
            b_value * u_value - coupling,
        )

    def reaction_rk4(u_value, v_value):
        k1u, k1v = reaction(u_value, v_value)
        k2u, k2v = reaction(u_value + 0.5 * dt * k1u, v_value + 0.5 * dt * k1v)
        k3u, k3v = reaction(u_value + 0.5 * dt * k2u, v_value + 0.5 * dt * k2v)
        k4u, k4v = reaction(u_value + dt * k3u, v_value + dt * k3v)
        return (
            u_value + dt * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0,
            v_value + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
        )

    snapshots: list[np.ndarray] = []
    times: list[float] = []
    saved_steps: list[int] = []
    if burn_steps == 0:
        _append_snapshot(snapshots, ops.module.stack((u, v)), ops, storage_dtype)
        times.append(0.0)
        saved_steps.append(0)

    total_steps = burn_steps + record_steps
    for step in _iter_steps(total_steps, progress, "Brusselator burn-in + solver steps"):
        u = ops.ifft2(diffuse_u * ops.fft2(u)).real
        v = ops.ifft2(diffuse_v * ops.fft2(v)).real
        u, v = reaction_rk4(u, v)
        u = ops.ifft2(diffuse_u * ops.fft2(u)).real
        v = ops.ifft2(diffuse_v * ops.fft2(v)).real
        relative_step = step - burn_steps
        if step >= burn_steps and (
            relative_step % save_every == 0 or relative_step == record_steps
        ):
            _append_snapshot(snapshots, ops.module.stack((u, v)), ops, storage_dtype)
            times.append(relative_step * dt)
            saved_steps.append(relative_step)

    ops.synchronize()
    return {
        "state": np.stack(snapshots),
        "time": np.asarray(times, dtype=np.float64),
        "step": np.asarray(saved_steps, dtype=np.int64),
        "x": x_np,
        "y": y_np,
    }


def solve_kolmogorov(
    config: dict[str, Any], seed: int, ops: ArrayBackend, progress: ProgressFactory | None
) -> dict[str, np.ndarray]:
    n = config["resolution"]
    length = config["domain_length"]
    dt = config["dt"]
    burn_steps = _steps(config["burn_in_time"], dt, "burn_in_time")
    record_steps = _steps(config["record_time"], dt, "record_time")
    save_every = config["save_every"]
    storage_dtype = np.dtype(config["storage_dtype"])
    reynolds = config["reynolds_number"]
    viscosity = 1.0 / reynolds
    forcing_amplitude = config["forcing_amplitude"]
    forcing_wavenumber = config["forcing_wavenumber"]

    x_np = np.linspace(0.0, length, n, endpoint=False, dtype=np.float64)
    y_np = np.linspace(0.0, length, n, endpoint=False, dtype=np.float64)
    _, yy_np = np.meshgrid(x_np, y_np, indexing="xy")
    dx = length / n
    wave_np = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx_np, ky_np = np.meshgrid(wave_np, wave_np, indexing="xy")
    k2_np = kx_np**2 + ky_np**2
    modes_np = np.fft.fftfreq(n) * n
    mx_np, my_np = np.meshgrid(modes_np, modes_np, indexing="xy")
    dealias_np = (np.abs(mx_np) <= n / 3) & (np.abs(my_np) <= n / 3)
    rng = np.random.default_rng(seed)
    raw_hat = np.fft.fft2(rng.standard_normal((n, n)))
    perturbation = np.fft.ifft2(
        raw_hat * np.exp(-config["perturbation_filter"] * k2_np**2) * dealias_np
    ).real
    perturbation -= perturbation.mean()
    perturbation /= perturbation.std()
    laminar = -(reynolds / forcing_wavenumber) * np.cos(forcing_wavenumber * yy_np)
    omega0_np = laminar + config["perturbation_amplitude"] * perturbation

    kx, ky, k2 = ops.asarray(kx_np), ops.asarray(ky_np), ops.asarray(k2_np)
    inverse_k2_np = np.zeros_like(k2_np)
    inverse_k2_np[k2_np > 0.0] = 1.0 / k2_np[k2_np > 0.0]
    inverse_k2 = ops.asarray(inverse_k2_np)
    dealias = ops.asarray(dealias_np.astype(np.float64)) > 0.5
    linear = -viscosity * k2
    vorticity_forcing_np = -forcing_amplitude * forcing_wavenumber * np.cos(
        forcing_wavenumber * yy_np
    )
    forcing_hat = ops.fft2(ops.asarray(vorticity_forcing_np))

    def velocity_from_vorticity(omega_hat_value):
        psi_hat = inverse_k2 * omega_hat_value
        u_value = ops.ifft2(1j * ky * psi_hat).real
        v_value = ops.ifft2(-1j * kx * psi_hat).real
        return u_value, v_value

    def nonlinear(omega_hat_value):
        u_value, v_value = velocity_from_vorticity(omega_hat_value)
        omega_x = ops.ifft2(1j * kx * omega_hat_value).real
        omega_y = ops.ifft2(1j * ky * omega_hat_value).real
        return (-ops.fft2(u_value * omega_x + v_value * omega_y) + forcing_hat) * dealias

    coefficients = _etdrk4_coefficients(ops, linear, dt, config["contour_points"])
    omega_hat = ops.fft2(ops.asarray(omega0_np)) * dealias
    snapshots: list[np.ndarray] = []
    times: list[float] = []
    saved_steps: list[int] = []
    total_steps = burn_steps + record_steps
    if burn_steps == 0:
        _append_snapshot(snapshots, ops.ifft2(omega_hat).real[None, :, :], ops, storage_dtype)
        times.append(0.0)
        saved_steps.append(0)

    for step in _iter_steps(total_steps, progress, "Kolmogorov burn-in + solver steps"):
        omega_hat = _etdrk4_step(omega_hat, nonlinear, coefficients) * dealias
        omega_hat[0, 0] = 0.0
        relative_step = step - burn_steps
        if step >= burn_steps and (
            relative_step % save_every == 0 or relative_step == record_steps
        ):
            _append_snapshot(
                snapshots, ops.ifft2(omega_hat).real[None, :, :], ops, storage_dtype
            )
            times.append(relative_step * dt)
            saved_steps.append(relative_step)

    ops.synchronize()
    return {
        "state": np.stack(snapshots),
        "time": np.asarray(times, dtype=np.float64),
        "step": np.asarray(saved_steps, dtype=np.int64),
        "x": x_np,
        "y": y_np,
    }


def _sobol_point(
    dimension: int,
    parameter_seed: int,
    parameter_index: int,
    lower: list[float],
    upper: list[float],
) -> np.ndarray:
    """Return one indexed point from a deterministic scrambled Sobol design."""

    if parameter_index < 0:
        raise ValueError("trajectory seed must not be smaller than seed_start")
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=parameter_seed)
    if parameter_index:
        sampler.fast_forward(parameter_index)
    return qmc.scale(sampler.random(1), lower, upper)[0].astype(np.float64)


def solve_electro_thermal(
    config: dict[str, Any], seed: int, ops: ArrayBackend, progress: ProgressFactory | None
) -> dict[str, np.ndarray]:
    """Solve one steady bidirectionally coupled electro-thermal realization."""

    if ops.name != "numpy":
        raise ValueError("electro_thermal requires the NumPy/SciPy CPU backend")
    n = config["resolution"]
    if n < 8:
        raise ValueError("electro_thermal resolution must be at least 8")
    length = config["domain_length"]
    storage_dtype = np.dtype(config["storage_dtype"])
    parameters = _sobol_point(
        5,
        config["parameter_seed"],
        seed,
        [
            config["ellipse_a_min"],
            config["ellipse_b_min"],
            config["ellipse_angle_min"],
            config["sigma_silicon_min"],
            config["kappa_alumina_min"],
        ],
        [
            config["ellipse_a_max"],
            config["ellipse_b_max"],
            config["ellipse_angle_max"],
            config["sigma_silicon_max"],
            config["kappa_alumina_max"],
        ],
    )
    a_value, b_value, phi, sigma_prefactor, kappa_alumina = parameters
    x = np.linspace(-length / 2.0, length / 2.0, n, dtype=np.float64)
    y = np.linspace(-length / 2.0, length / 2.0, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    dx, dy = x[1] - x[0], y[1] - y[0]
    cosine, sine = np.cos(phi), np.sin(phi)
    rotated_x = cosine * xx + sine * yy
    rotated_y = -sine * xx + cosine * yy
    ellipse_mask = (rotated_x / a_value) ** 2 + (rotated_y / b_value) ** 2 <= 1.0

    absorbing_thickness = config["absorbing_thickness"]
    pml_cells = max(2, int(round(absorbing_thickness / dx)))
    full_n = n + 2 * pml_cells
    x_full = (np.arange(full_n) - (full_n - 1) / 2.0) * dx
    y_full = (np.arange(full_n) - (full_n - 1) / 2.0) * dy
    xx_full, yy_full = np.meshgrid(x_full, y_full, indexing="xy")
    physical_slice = np.s_[pml_cells : pml_cells + n, pml_cells : pml_cells + n]
    omega = 2.0 * np.pi * config["frequency"]
    epsilon_0 = 8.8541878128e-12
    k0 = omega / 299792458.0

    def silicon_conductivity(temperature):
        if np.any(temperature <= 0.0):
            raise FloatingPointError("nonpositive absolute temperature in conductivity law")
        return 1.602 * sigma_prefactor * np.exp(
            -1.12 / (8.6173e-5 * temperature)
        )

    def coordinate_stretch(coordinates):
        depth = np.maximum(np.abs(coordinates) - length / 2.0, 0.0)
        normalized_depth = depth / absorbing_thickness
        return 1.0 - 1j * config["pml_strength"] * normalized_depth ** config["pml_power"]

    sx = coordinate_stretch(x_full)
    sy = coordinate_stretch(y_full)
    sx_face = 0.5 * (sx[:-1] + sx[1:])
    sy_face = 0.5 * (sy[:-1] + sy[1:])
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    electric_boundary = np.zeros((full_n, full_n), dtype=bool)
    electric_boundary[[0, -1], :] = True
    electric_boundary[:, [0, -1]] = True
    for row in range(full_n):
        for column in range(full_n):
            index = row * full_n + column
            if electric_boundary[row, column]:
                rows.append(index); columns.append(index); values.append(1.0 + 0.0j)
                continue
            west = 1.0 / (sx[column] * sx_face[column - 1] * dx**2)
            east = 1.0 / (sx[column] * sx_face[column] * dx**2)
            south = 1.0 / (sy[row] * sy_face[row - 1] * dy**2)
            north = 1.0 / (sy[row] * sy_face[row] * dy**2)
            rows.extend((index, index, index, index, index))
            columns.extend((index, index - 1, index + 1, index - full_n, index + full_n))
            values.extend((-(west + east + south + north), west, east, south, north))
    pml_laplacian = sp.csr_matrix(
        (values, (rows, columns)), shape=(full_n**2, full_n**2)
    )
    incident = config["incident_amplitude"] * np.exp(
        1j
        * k0
        * (
            -xx_full * np.cos(config["incident_angle"])
            - yy_full * np.sin(config["incident_angle"])
        )
    )

    def solve_electric(temperature):
        epsilon_relative = np.full(
            (full_n, full_n), config["permittivity_alumina"], dtype=np.float64
        )
        sigma = np.full(
            (full_n, full_n), config["conductivity_alumina"], dtype=np.float64
        )
        epsilon_relative[physical_slice] = np.where(
            ellipse_mask,
            config["permittivity_silicon"],
            config["permittivity_alumina"],
        )
        sigma_crop = np.where(
            ellipse_mask,
            silicon_conductivity(temperature),
            config["conductivity_alumina"],
        )
        sigma[physical_slice] = sigma_crop
        potential = k0**2 * (
            epsilon_relative - 1j * sigma / (omega * epsilon_0)
        )
        potential[electric_boundary] = 0.0
        operator = pml_laplacian + sp.diags(potential.ravel(), format="csr")
        right_hand_side = -(operator @ incident.ravel())
        right_hand_side[electric_boundary.ravel()] = 0.0
        scattered = spla.spsolve(operator.tocsc(), right_hand_side).reshape(full_n, full_n)
        total = incident + scattered
        if not np.isfinite(total).all():
            raise FloatingPointError("non-finite electro-thermal Helmholtz solution")
        return total[physical_slice], sigma_crop

    def harmonic_mean(first, second):
        return 2.0 * first * second / (first + second + 1.0e-300)

    thermal_conductivity = np.where(
        ellipse_mask, config["thermal_conductivity_silicon"], kappa_alumina
    )
    rows, columns, values = [], [], []
    robin_rhs = np.zeros(n * n, dtype=np.float64)
    heat_boundary = np.zeros((n, n), dtype=bool)
    heat_boundary[[0, -1], :] = True
    heat_boundary[:, [0, -1]] = True
    for row in range(n):
        for column in range(n):
            index = row * n + column
            if heat_boundary[row, column]:
                diagonal = 0.0
                boundary_count = 0
                if column == 0:
                    coefficient = thermal_conductivity[row, column] / dx
                    diagonal += coefficient; boundary_count += 1
                    rows.append(index); columns.append(index + 1); values.append(-coefficient)
                if column == n - 1:
                    coefficient = thermal_conductivity[row, column] / dx
                    diagonal += coefficient; boundary_count += 1
                    rows.append(index); columns.append(index - 1); values.append(-coefficient)
                if row == 0:
                    coefficient = thermal_conductivity[row, column] / dy
                    diagonal += coefficient; boundary_count += 1
                    rows.append(index); columns.append(index + n); values.append(-coefficient)
                if row == n - 1:
                    coefficient = thermal_conductivity[row, column] / dy
                    diagonal += coefficient; boundary_count += 1
                    rows.append(index); columns.append(index - n); values.append(-coefficient)
                diagonal += boundary_count * config["convective_coefficient"]
                robin_rhs[index] = (
                    boundary_count
                    * config["convective_coefficient"]
                    * config["ambient_temperature"]
                )
                rows.append(index); columns.append(index); values.append(diagonal)
                continue
            west = harmonic_mean(
                thermal_conductivity[row, column], thermal_conductivity[row, column - 1]
            ) / dx**2
            east = harmonic_mean(
                thermal_conductivity[row, column], thermal_conductivity[row, column + 1]
            ) / dx**2
            south = harmonic_mean(
                thermal_conductivity[row, column], thermal_conductivity[row - 1, column]
            ) / dy**2
            north = harmonic_mean(
                thermal_conductivity[row, column], thermal_conductivity[row + 1, column]
            ) / dy**2
            rows.extend((index, index, index, index, index))
            columns.extend((index, index - 1, index + 1, index - n, index + n))
            values.extend((west + east + south + north, -west, -east, -south, -north))
    heat_operator = sp.csr_matrix((values, (rows, columns)), shape=(n * n, n * n))
    heat_solve = spla.factorized(heat_operator.tocsc())

    temperature = np.full(
        (n, n), config["ambient_temperature"], dtype=np.float64
    )
    sigma_previous = np.where(
        ellipse_mask,
        silicon_conductivity(temperature),
        config["conductivity_alumina"],
    )
    temperature_update = np.inf
    conductivity_update = np.inf
    iteration_values = _iter_steps(
        config["maximum_coupling_iterations"],
        progress,
        "Electro-thermal Picard coupling",
    )
    for iteration in iteration_values:
        electric, _ = solve_electric(temperature)
        sigma_current = np.where(
            ellipse_mask,
            silicon_conductivity(temperature),
            config["conductivity_alumina"],
        )
        joule_heating = 0.5 * sigma_current * np.abs(electric) ** 2
        right_hand_side = joule_heating.ravel().copy()
        right_hand_side[heat_boundary.ravel()] = robin_rhs[heat_boundary.ravel()]
        candidate = heat_solve(right_hand_side).reshape(n, n)
        relaxed = temperature + config["under_relaxation"] * (candidate - temperature)
        sigma_next = np.where(
            ellipse_mask,
            silicon_conductivity(relaxed),
            config["conductivity_alumina"],
        )
        temperature_update = np.max(np.abs(relaxed - temperature)) / (
            np.max(np.abs(relaxed)) + 1.0e-30
        )
        conductivity_update = np.max(np.abs(sigma_next - sigma_previous)) / (
            np.max(np.abs(sigma_next)) + 1.0e-30
        )
        temperature = relaxed
        sigma_previous = sigma_next
        if max(temperature_update, conductivity_update) < config["coupling_tolerance"]:
            break
    else:
        raise RuntimeError(
            "electro-thermal Picard coupling failed after "
            f"{config['maximum_coupling_iterations']} iterations"
        )
    electric, sigma_final = solve_electric(temperature)
    joule_heating = 0.5 * sigma_final * np.abs(electric) ** 2
    if temperature.min() < config["ambient_temperature"] - 1.0e-7:
        raise RuntimeError("temperature dropped below ambient beyond solver tolerance")

    def physical_laplacian(field):
        return (
            (field[2:-2, 3:-1] - 2.0 * field[2:-2, 2:-2] + field[2:-2, 1:-3]) / dx**2
            + (field[3:-1, 2:-2] - 2.0 * field[2:-2, 2:-2] + field[1:-3, 2:-2]) / dy**2
        )

    epsilon_crop = np.where(
        ellipse_mask,
        config["permittivity_silicon"],
        config["permittivity_alumina"],
    )
    laplace_term = physical_laplacian(electric)
    potential_term = (
        k0**2
        * (
            epsilon_crop[2:-2, 2:-2]
            - 1j * sigma_final[2:-2, 2:-2] / (omega * epsilon_0)
        )
        * electric[2:-2, 2:-2]
    )
    helmholtz_residual = np.linalg.norm(laplace_term + potential_term) / (
        np.linalg.norm(laplace_term) + np.linalg.norm(potential_term) + 1.0e-30
    )
    heat_rhs = joule_heating.ravel().copy()
    heat_rhs[heat_boundary.ravel()] = robin_rhs[heat_boundary.ravel()]
    heat_left = (heat_operator @ temperature.ravel()).reshape(n, n)
    heat_error = (heat_left.ravel() - heat_rhs).reshape(n, n)[2:-2, 2:-2]
    heat_residual = np.linalg.norm(heat_error) / (
        np.linalg.norm(heat_left[2:-2, 2:-2])
        + np.linalg.norm(joule_heating[2:-2, 2:-2])
        + 1.0e-30
    )
    boundary_flux = np.concatenate(
        (
            thermal_conductivity[:, 0] * (temperature[:, 1] - temperature[:, 0]) / dx,
            thermal_conductivity[:, -1] * (temperature[:, -2] - temperature[:, -1]) / dx,
            thermal_conductivity[0, :] * (temperature[1, :] - temperature[0, :]) / dy,
            thermal_conductivity[-1, :] * (temperature[-2, :] - temperature[-1, :]) / dy,
        )
    )
    boundary_robin = config["convective_coefficient"] * np.concatenate(
        (
            temperature[:, 0] - config["ambient_temperature"],
            temperature[:, -1] - config["ambient_temperature"],
            temperature[0, :] - config["ambient_temperature"],
            temperature[-1, :] - config["ambient_temperature"],
        )
    )
    robin_residual = np.linalg.norm(boundary_flux - boundary_robin) / (
        np.linalg.norm(boundary_flux) + np.linalg.norm(boundary_robin) + 1.0e-30
    )
    fields = np.stack((electric.real, electric.imag, temperature))[None]
    return {
        "state": fields.astype(storage_dtype, copy=False),
        "time": np.asarray([0.0], dtype=np.float64),
        "step": np.asarray([0], dtype=np.int64),
        "x": x,
        "y": y,
        "condition_values": parameters,
        "ellipse_mask": ellipse_mask.astype(np.uint8),
        "conductivity": sigma_final.astype(storage_dtype, copy=False),
        "joule_heating": joule_heating.astype(storage_dtype, copy=False),
        "thermal_conductivity": thermal_conductivity.astype(storage_dtype, copy=False),
        "coupling_iterations": np.asarray(iteration, dtype=np.int64),
        "relative_temperature_update": np.asarray(temperature_update, dtype=np.float64),
        "relative_conductivity_update": np.asarray(conductivity_update, dtype=np.float64),
        "relative_helmholtz_residual": np.asarray(helmholtz_residual, dtype=np.float64),
        "relative_heat_residual": np.asarray(heat_residual, dtype=np.float64),
        "relative_robin_boundary_residual": np.asarray(robin_residual, dtype=np.float64),
    }


def solve_mass_transport_fluid(
    config: dict[str, Any], seed: int, ops: ArrayBackend, progress: ProgressFactory | None
) -> dict[str, np.ndarray]:
    """Solve one transient density-coupled Elder-type realization."""

    if ops.name != "numpy":
        raise ValueError("mass_transport_fluid requires the NumPy/SciPy CPU backend")
    n = config["resolution"]
    if n < 8:
        raise ValueError("mass_transport_fluid resolution must be at least 8")
    storage_dtype = np.dtype(config["storage_dtype"])
    parameters = _sobol_point(
        4,
        config["parameter_seed"],
        seed,
        [
            config["source_amplitude_min"],
            config["source_x_min"],
            config["source_y_min"],
            config["source_width_min"],
        ],
        [
            config["source_amplitude_max"],
            config["source_x_max"],
            config["source_y_max"],
            config["source_width_max"],
        ],
    )
    amplitude, source_x, source_y, source_width = parameters
    length = config["domain_length"]
    height = config["domain_height"]
    x = np.linspace(-length / 2.0, length / 2.0, n, dtype=np.float64)
    y = np.linspace(-height / 2.0, height / 2.0, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    dx, dy = x[1] - x[0], y[1] - y[0]
    seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
    gravity_y = -config["gravity"]
    mobility = config["permeability"] / config["dynamic_viscosity"]
    source = amplitude / seconds_per_year * np.exp(
        -((xx - source_x) ** 2 + (yy - source_y) ** 2) / (2.0 * source_width**2)
    )
    bottom_dirichlet = np.zeros((n, n), dtype=bool)
    bottom_dirichlet[0, :] = True
    top_right_dirichlet = np.zeros((n, n), dtype=bool)
    top_right_dirichlet[-1, x >= 0.0] = True
    concentration_dirichlet = bottom_dirichlet | top_right_dirichlet
    boundary_values = np.zeros((n, n), dtype=np.float64)
    boundary_values[top_right_dirichlet] = config["surface_concentration"]

    def density_from_concentration(concentration):
        return config["rho0"] + config["density_coefficient"] * np.maximum(
            concentration, 0.0
        )

    def build_pressure_system(density, density_old, dt_seconds):
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        gravity_mass_y = np.zeros((n + 1, n), dtype=np.float64)
        rho_x = 0.5 * (density[:, :-1] + density[:, 1:])
        rho_y = 0.5 * (density[:-1, :] + density[1:, :])
        gravity_mass_y[1:-1, :] = mobility * rho_y**2 * gravity_y
        for row in range(n):
            for column in range(n):
                index = row * n + column
                diagonal = 0.0
                if column > 0:
                    coefficient = mobility * rho_x[row, column - 1] / dx**2
                    diagonal += coefficient
                    rows.append(index); columns.append(index - 1); values.append(-coefficient)
                if column < n - 1:
                    coefficient = mobility * rho_x[row, column] / dx**2
                    diagonal += coefficient
                    rows.append(index); columns.append(index + 1); values.append(-coefficient)
                if row > 0:
                    coefficient = mobility * rho_y[row - 1, column] / dy**2
                    diagonal += coefficient
                    rows.append(index); columns.append(index - n); values.append(-coefficient)
                if row < n - 1:
                    coefficient = mobility * rho_y[row, column] / dy**2
                    diagonal += coefficient
                    rows.append(index); columns.append(index + n); values.append(-coefficient)
                rows.append(index); columns.append(index); values.append(diagonal)
        operator = sp.csr_matrix((values, (rows, columns)), shape=(n * n, n * n))
        gravity_divergence = (
            gravity_mass_y[1:, :] - gravity_mass_y[:-1, :]
        ) / dy
        unprojected_rhs = (
            -config["porosity"] * (density - density_old) / dt_seconds
            - gravity_divergence
        )
        compatibility_mismatch = float(unprojected_rhs.mean())
        right_hand_side = (unprojected_rhs - compatibility_mismatch).ravel()
        operator = operator.tolil()
        operator[0, :] = 0.0
        operator[0, 0] = 1.0
        right_hand_side[0] = 0.0
        return operator.tocsc(), right_hand_side, compatibility_mismatch

    def darcy_face_fluxes(pressure, density):
        velocity_x = np.zeros((n, n + 1), dtype=np.float64)
        velocity_y = np.zeros((n + 1, n), dtype=np.float64)
        rho_x = 0.5 * (density[:, :-1] + density[:, 1:])
        rho_y = 0.5 * (density[:-1, :] + density[1:, :])
        velocity_x[:, 1:-1] = -mobility * (pressure[:, 1:] - pressure[:, :-1]) / dx
        velocity_y[1:-1, :] = -mobility * (
            (pressure[1:, :] - pressure[:-1, :]) / dy - rho_y * gravity_y
        )
        mass_flux_x = np.zeros_like(velocity_x)
        mass_flux_y = np.zeros_like(velocity_y)
        mass_flux_x[:, 1:-1] = rho_x * velocity_x[:, 1:-1]
        mass_flux_y[1:-1, :] = rho_y * velocity_y[1:-1, :]
        return velocity_x, velocity_y, mass_flux_x, mass_flux_y

    def cell_velocity(velocity_x, velocity_y):
        return (
            0.5 * (velocity_x[:, :-1] + velocity_x[:, 1:]),
            0.5 * (velocity_y[:-1, :] + velocity_y[1:, :]),
        )

    def upwind_flux(concentration, velocity_x, velocity_y):
        flux_x = np.zeros_like(velocity_x)
        flux_y = np.zeros_like(velocity_y)
        flux_x[:, 1:-1] = velocity_x[:, 1:-1] * np.where(
            velocity_x[:, 1:-1] >= 0.0,
            concentration[:, :-1],
            concentration[:, 1:],
        )
        flux_y[1:-1, :] = velocity_y[1:-1, :] * np.where(
            velocity_y[1:-1, :] >= 0.0,
            concentration[:-1, :],
            concentration[1:, :],
        )
        return flux_x, flux_y

    transport_solvers: dict[float, Callable[[np.ndarray], np.ndarray]] = {}

    def transport_solve_for_step(dt_seconds):
        key = float(dt_seconds)
        if key in transport_solvers:
            return transport_solvers[key]
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        transient = config["porosity"] / dt_seconds
        diffusion_x = config["porosity"] * config["diffusivity"] / dx**2
        diffusion_y = config["porosity"] * config["diffusivity"] / dy**2
        for row in range(n):
            for column in range(n):
                index = row * n + column
                if concentration_dirichlet[row, column]:
                    rows.append(index); columns.append(index); values.append(1.0)
                    continue
                diagonal = transient
                if column > 0:
                    diagonal += diffusion_x
                    rows.append(index); columns.append(index - 1); values.append(-diffusion_x)
                if column < n - 1:
                    diagonal += diffusion_x
                    rows.append(index); columns.append(index + 1); values.append(-diffusion_x)
                if row > 0:
                    diagonal += diffusion_y
                    rows.append(index); columns.append(index - n); values.append(-diffusion_y)
                if row < n - 1:
                    diagonal += diffusion_y
                    rows.append(index); columns.append(index + n); values.append(-diffusion_y)
                rows.append(index); columns.append(index); values.append(diagonal)
        operator = sp.csc_matrix((values, (rows, columns)), shape=(n * n, n * n))
        transport_solvers[key] = spla.factorized(operator)
        return transport_solvers[key]

    def solve_transport(
        concentration_old, concentration_for_flux, velocity_x, velocity_y, dt_seconds
    ):
        flux_x, flux_y = upwind_flux(concentration_for_flux, velocity_x, velocity_y)
        advective_divergence = (
            (flux_x[:, 1:] - flux_x[:, :-1]) / dx
            + (flux_y[1:, :] - flux_y[:-1, :]) / dy
        )
        rhs = (
            config["porosity"] / dt_seconds * concentration_old
            + source
            - advective_divergence
        ).ravel()
        rhs[concentration_dirichlet.ravel()] = boundary_values.ravel()[
            concentration_dirichlet.ravel()
        ]
        return transport_solve_for_step(dt_seconds)(rhs).reshape(n, n)

    def coupled_step(concentration_old, dt_seconds):
        concentration = concentration_old.copy()
        density_old = density_from_concentration(concentration_old)
        minimum_before_cleanup = np.inf
        compatibility_mismatch = 0.0
        for iteration in range(1, config["maximum_picard_iterations"] + 1):
            density = density_from_concentration(concentration)
            pressure_operator, pressure_rhs, compatibility_mismatch = build_pressure_system(
                density, density_old, dt_seconds
            )
            pressure = spla.spsolve(pressure_operator, pressure_rhs).reshape(n, n)
            pressure -= pressure.mean()
            velocity_x, velocity_y, _, _ = darcy_face_fluxes(pressure, density)
            candidate = solve_transport(
                concentration_old, concentration, velocity_x, velocity_y, dt_seconds
            )
            minimum_before_cleanup = min(minimum_before_cleanup, float(candidate.min()))
            if not np.isfinite(candidate).all() or candidate.min() < -1.0e-7:
                return None
            relaxed = concentration + config["picard_relaxation"] * (
                candidate - concentration
            )
            relaxed[concentration_dirichlet] = boundary_values[concentration_dirichlet]
            concentration_update = np.max(np.abs(relaxed - concentration)) / (
                max(np.max(np.abs(relaxed)), config["surface_concentration"]) + 1.0e-30
            )
            density_update = np.max(
                np.abs(density_from_concentration(relaxed) - density)
            ) / (np.max(density_from_concentration(relaxed)) + 1.0e-30)
            concentration = relaxed
            if max(concentration_update, density_update) < config["picard_tolerance"]:
                break
        else:
            return None
        density = density_from_concentration(concentration)
        pressure_operator, pressure_rhs, compatibility_mismatch = build_pressure_system(
            density, density_old, dt_seconds
        )
        pressure = spla.spsolve(pressure_operator, pressure_rhs).reshape(n, n)
        pressure -= pressure.mean()
        velocity_x, velocity_y, mass_flux_x, mass_flux_y = darcy_face_fluxes(
            pressure, density
        )
        ux, uy = cell_velocity(velocity_x, velocity_y)
        courant = float(
            np.sqrt(ux**2 + uy**2).max()
            * dt_seconds
            / (config["porosity"] * min(dx, dy))
        )
        if courant > config["advective_cfl"] * (1.0 + 1.0e-10):
            return None
        concentration[
            (concentration < 0.0) & (concentration >= -1.0e-7)
        ] = 0.0
        return {
            "concentration": concentration,
            "pressure": pressure,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "mass_flux_x": mass_flux_x,
            "mass_flux_y": mass_flux_y,
            "iterations": iteration,
            "courant": courant,
            "minimum_before_cleanup": minimum_before_cleanup,
            "compatibility_mismatch": compatibility_mismatch,
        }

    output_interval = config["dt"] * config["save_every"]
    output_count = int(round(config["record_time"] / output_interval))
    times = np.arange(output_count + 1, dtype=np.float64) * output_interval
    concentration = np.zeros((n, n), dtype=np.float64)
    concentration[concentration_dirichlet] = boundary_values[concentration_dirichlet]
    pressure = config["rho0"] * gravity_y * yy
    pressure -= pressure.mean()
    velocity_x = np.zeros((n, n + 1), dtype=np.float64)
    velocity_y = np.zeros((n + 1, n), dtype=np.float64)
    ux, uy = cell_velocity(velocity_x, velocity_y)
    frames = [np.stack((ux, uy, concentration))]
    pressure_frames = [pressure.copy()]
    velocity_x_frames = [velocity_x.copy()]
    velocity_y_frames = [velocity_y.copy()]
    minimum_before_cleanup = float(concentration.min())
    iteration_counts: list[int] = []
    compatibility_mismatches: list[float] = []
    reduced_steps = 0
    rejected_steps = 0
    current_time = 0.0
    proposed_step = config["dt"]
    target_indices = _iter_steps(
        output_count, progress, "Mass-transport output intervals"
    )
    for target_index in target_indices:
        target_time = times[target_index]
        while current_time < target_time - 1.0e-12:
            step_years = min(proposed_step, target_time - current_time)
            while True:
                result = coupled_step(concentration, step_years * seconds_per_year)
                if result is not None:
                    break
                rejected_steps += 1
                reduced_steps += 1
                step_years *= 0.5
                if step_years < config["minimum_step_years"]:
                    raise RuntimeError(
                        "adaptive mass-transport step fell below minimum after "
                        "nonlinear/CFL rejection"
                    )
            if step_years < proposed_step - 1.0e-14:
                reduced_steps += 1
            concentration = result["concentration"]
            pressure = result["pressure"]
            velocity_x = result["velocity_x"]
            velocity_y = result["velocity_y"]
            current_time += step_years
            minimum_before_cleanup = min(
                minimum_before_cleanup, result["minimum_before_cleanup"]
            )
            iteration_counts.append(result["iterations"])
            compatibility_mismatches.append(result["compatibility_mismatch"])
            if result["courant"] < 0.55 * config["advective_cfl"]:
                proposed_step = min(config["dt"], step_years * 1.35)
            else:
                proposed_step = step_years
        ux, uy = cell_velocity(velocity_x, velocity_y)
        frames.append(np.stack((ux, uy, concentration)))
        pressure_frames.append(pressure.copy())
        velocity_x_frames.append(velocity_x.copy())
        velocity_y_frames.append(velocity_y.copy())

    return {
        "state": np.asarray(frames, dtype=storage_dtype),
        "time": times,
        "step": np.rint(times / config["dt"]).astype(np.int64),
        "x": x,
        "y": y,
        "condition_values": parameters,
        "source_field": source.astype(storage_dtype, copy=False),
        "pressure": np.asarray(pressure_frames, dtype=storage_dtype),
        "velocity_x_face": np.asarray(velocity_x_frames, dtype=storage_dtype),
        "velocity_y_face": np.asarray(velocity_y_frames, dtype=storage_dtype),
        "picard_iterations": np.asarray(iteration_counts, dtype=np.int64),
        "compatibility_mismatches": np.asarray(
            compatibility_mismatches, dtype=np.float64
        ),
        "minimum_concentration_before_cleanup": np.asarray(
            minimum_before_cleanup, dtype=np.float64
        ),
        "reduced_timestep_count": np.asarray(reduced_steps, dtype=np.int64),
        "rejected_timestep_count": np.asarray(rejected_steps, dtype=np.int64),
    }


SOLVERS = {
    "burgers": solve_burgers,
    "ks": solve_ks,
    "brusselator": solve_brusselator,
    "kolmogorov": solve_kolmogorov,
    "electro_thermal": solve_electro_thermal,
    "mass_transport_fluid": solve_mass_transport_fluid,
}


def run_solver(
    case: str,
    config: dict[str, Any],
    seed: int,
    progress: ProgressFactory | None = None,
) -> tuple[dict[str, np.ndarray], ArrayBackend]:
    ops = ArrayBackend(config["backend"], config["device"], config["solver_dtype"])
    return SOLVERS[case](config, seed, ops, progress), ops

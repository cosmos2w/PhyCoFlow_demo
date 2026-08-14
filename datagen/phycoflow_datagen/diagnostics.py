"""Physical diagnostics and derived fields shared by processing and plotting."""

from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1.0e-12


def _relative_norm(residuals: list[np.ndarray], references: list[np.ndarray]) -> float:
    if not residuals:
        return float("nan")
    return float(np.linalg.norm(np.asarray(residuals)) / (np.linalg.norm(np.asarray(references)) + EPS))


def _grid_1d(n: int, length: float):
    dx = length / n
    wave = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    modes = np.fft.fftfreq(n) * n
    return wave, np.abs(modes) <= n / 3


def _grid_2d(n: int, length: float):
    dx = length / n
    wave = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(wave, wave, indexing="xy")
    k2 = kx**2 + ky**2
    modes = np.fft.fftfreq(n) * n
    mx, my = np.meshgrid(modes, modes, indexing="xy")
    dealias = (np.abs(mx) <= n / 3) & (np.abs(my) <= n / 3)
    return kx, ky, k2, dealias


def derive_kolmogorov_fields(
    omega_state: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    """Convert saved vorticity to canonical `(u, v, p)` fields.

    Input is `[T,1,H,W]` or `[T,H,W]`; output is `[T,3,H,W]`.
    Pressure uses the periodic zero-spatial-mean gauge from the notebook.
    """

    omega = np.asarray(omega_state)
    if omega.ndim == 4:
        if omega.shape[1] != 1:
            raise ValueError(f"expected one vorticity channel, got shape {omega.shape}")
        omega = omega[:, 0]
    if omega.ndim != 3 or omega.shape[-1] != omega.shape[-2]:
        raise ValueError(f"expected vorticity [T,H,W] on a square grid, got {omega.shape}")

    n = omega.shape[-1]
    kx, ky, k2, dealias = _grid_2d(n, config["domain_length"])
    inverse_k2 = np.zeros_like(k2)
    inverse_k2[k2 > 0.0] = 1.0 / k2[k2 > 0.0]
    frames = []
    for omega_frame in omega:
        omega_hat = np.fft.fft2(omega_frame)
        psi_hat = inverse_k2 * omega_hat
        u = np.fft.ifft2(1j * ky * psi_hat).real
        v = np.fft.ifft2(-1j * kx * psi_hat).real
        ux = np.fft.ifft2(1j * kx * np.fft.fft2(u)).real
        uy = np.fft.ifft2(1j * ky * np.fft.fft2(u)).real
        vx = np.fft.ifft2(1j * kx * np.fft.fft2(v)).real
        vy = np.fft.ifft2(1j * ky * np.fft.fft2(v)).real
        adv_u = np.fft.ifft2(np.fft.fft2(u * ux + v * uy) * dealias).real
        adv_v = np.fft.ifft2(np.fft.fft2(u * vx + v * vy) * dealias).real
        divergence_hat = 1j * kx * np.fft.fft2(adv_u) + 1j * ky * np.fft.fft2(adv_v)
        pressure_hat = divergence_hat * inverse_k2 * dealias
        pressure_hat[0, 0] = 0.0
        pressure = np.fft.ifft2(pressure_hat).real
        frames.append(np.stack((u, v, pressure)))
    return np.asarray(frames)


def vorticity_from_velocity(fields: np.ndarray, domain_length: float) -> np.ndarray:
    n = fields.shape[-1]
    kx, ky, _, _ = _grid_2d(n, domain_length)
    u_hat = np.fft.fft2(fields[:, 0], axes=(-2, -1))
    v_hat = np.fft.fft2(fields[:, 1], axes=(-2, -1))
    return np.fft.ifft2(1j * kx * v_hat - 1j * ky * u_hat, axes=(-2, -1)).real


def burgers_diagnostics(state: np.ndarray, time: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    u = np.asarray(state)[:, 0]
    wave, dealias = _grid_1d(u.shape[-1], config["domain_length"])
    viscosity = config["viscosity"]
    residuals, time_terms = [], []
    for index in range(1, time.size - 1):
        ut = (u[index + 1] - u[index - 1]) / (time[index + 1] - time[index - 1])
        advection = np.fft.ifft(0.5j * wave * np.fft.fft(u[index] ** 2) * dealias).real
        diffusion = np.fft.ifft(-viscosity * wave**2 * np.fft.fft(u[index])).real
        residuals.append(ut + advection - diffusion)
        time_terms.append(ut)
    gradients = np.fft.ifft(1j * wave * np.fft.fft(u, axis=-1), axis=-1).real
    energy = 0.5 * np.mean(u**2, axis=-1)
    return {
        "finite": bool(np.isfinite(u).all()),
        "relative_pde_residual": _relative_norm(residuals, time_terms),
        "maximum_abs_gradient": float(np.max(np.abs(gradients))),
        "maximum_mean_drift": float(np.max(np.abs(u.mean(axis=-1) - u[0].mean()))),
        "initial_energy": float(energy[0]),
        "final_energy": float(energy[-1]),
        "maximum_energy_increase": float(max(0.0, np.max(np.diff(energy), initial=0.0))),
    }


def ks_diagnostics(state: np.ndarray, time: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    u = np.asarray(state)[:, 0]
    wave, dealias = _grid_1d(u.shape[-1], config["domain_length"])
    residuals, time_terms = [], []
    for index in range(1, time.size - 1):
        ut = (u[index + 1] - u[index - 1]) / (time[index + 1] - time[index - 1])
        field_hat = np.fft.fft(u[index])
        advection = np.fft.ifft(
            0.5j
            * config["advection_coefficient"]
            * wave
            * np.fft.fft(u[index] ** 2)
            * dealias
        ).real
        uxx = config["second_order_coefficient"] * np.fft.ifft(
            -(wave**2) * field_hat
        ).real
        uxxxx = config["fourth_order_coefficient"] * np.fft.ifft(
            wave**4 * field_hat
        ).real
        residuals.append(ut + advection + uxx + uxxxx)
        time_terms.append(ut)
    return {
        "finite": bool(np.isfinite(u).all()),
        "relative_pde_residual": _relative_norm(residuals, time_terms),
        "maximum_mean_drift": float(np.max(np.abs(u.mean(axis=-1) - u[0].mean()))),
        "initial_std": float(u[0].std()),
        "final_std": float(u[-1].std()),
        "mean_temporal_std": float(u.std(axis=0).mean()),
    }


def brusselator_diagnostics(
    state: np.ndarray, time: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    fields = np.asarray(state)
    u, v = fields[:, 0], fields[:, 1]
    _, _, k2, _ = _grid_2d(u.shape[-1], config["domain_length"])

    def laplacian(field):
        return np.fft.ifft2(-k2 * np.fft.fft2(field)).real

    residual_u, residual_v, time_u, time_v = [], [], [], []
    for index in range(1, time.size - 1):
        ut = (u[index + 1] - u[index - 1]) / (time[index + 1] - time[index - 1])
        vt = (v[index + 1] - v[index - 1]) / (time[index + 1] - time[index - 1])
        coupling = u[index] ** 2 * v[index]
        reaction_u = config["A"] - (config["B"] + 1.0) * u[index] + coupling
        reaction_v = config["B"] * u[index] - coupling
        residual_u.append(ut - config["diffusivity_u"] * laplacian(u[index]) - reaction_u)
        residual_v.append(vt - config["diffusivity_v"] * laplacian(v[index]) - reaction_v)
        time_u.append(ut)
        time_v.append(vt)
    return {
        "finite": bool(np.isfinite(fields).all()),
        "relative_pde_residual_u": _relative_norm(residual_u, time_u),
        "relative_pde_residual_v": _relative_norm(residual_v, time_v),
        "minimum_concentration": float(fields.min()),
        "maximum_concentration": float(fields.max()),
        "final_uv_correlation": float(np.corrcoef(u[-1].ravel(), v[-1].ravel())[0, 1]),
    }


def kolmogorov_diagnostics(
    state: np.ndarray, time: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    omega = np.asarray(state)[:, 0]
    fields = derive_kolmogorov_fields(state, config)
    u_all, v_all, p_all = fields[:, 0], fields[:, 1], fields[:, 2]
    n = omega.shape[-1]
    kx, ky, k2, dealias = _grid_2d(n, config["domain_length"])
    viscosity = 1.0 / config["reynolds_number"]
    y = np.linspace(0.0, config["domain_length"], n, endpoint=False)
    _, yy = np.meshgrid(y, y, indexing="xy")
    force_x = config["forcing_amplitude"] * np.sin(config["forcing_wavenumber"] * yy)
    divergence_rms = []
    for u, v in zip(u_all, v_all):
        divergence = np.fft.ifft2(1j * kx * np.fft.fft2(u) + 1j * ky * np.fft.fft2(v)).real
        divergence_rms.append(np.sqrt(np.mean(divergence**2)))

    residual_u, residual_v, time_u, time_v = [], [], [], []
    for index in range(1, time.size - 1):
        u, v, p = fields[index]
        ut = (u_all[index + 1] - u_all[index - 1]) / (time[index + 1] - time[index - 1])
        vt = (v_all[index + 1] - v_all[index - 1]) / (time[index + 1] - time[index - 1])
        ux = np.fft.ifft2(1j * kx * np.fft.fft2(u)).real
        uy = np.fft.ifft2(1j * ky * np.fft.fft2(u)).real
        vx = np.fft.ifft2(1j * kx * np.fft.fft2(v)).real
        vy = np.fft.ifft2(1j * ky * np.fft.fft2(v)).real
        px = np.fft.ifft2(1j * kx * np.fft.fft2(p)).real
        py = np.fft.ifft2(1j * ky * np.fft.fft2(p)).real
        adv_u = np.fft.ifft2(np.fft.fft2(u * ux + v * uy) * dealias).real
        adv_v = np.fft.ifft2(np.fft.fft2(u * vx + v * vy) * dealias).real
        lap_u = np.fft.ifft2(-k2 * np.fft.fft2(u)).real
        lap_v = np.fft.ifft2(-k2 * np.fft.fft2(v)).real
        residual_u.append(ut + adv_u + px - viscosity * lap_u - force_x)
        residual_v.append(vt + adv_v + py - viscosity * lap_v)
        time_u.append(ut)
        time_v.append(vt)

    energy = 0.5 * np.mean(u_all**2 + v_all**2, axis=(-2, -1))
    enstrophy = 0.5 * np.mean(omega**2, axis=(-2, -1))
    return {
        "finite": bool(np.isfinite(fields).all()),
        "relative_momentum_residual_u": _relative_norm(residual_u, time_u),
        "relative_momentum_residual_v": _relative_norm(residual_v, time_v),
        "maximum_divergence_rms": float(np.max(divergence_rms)),
        "maximum_abs_pressure_mean": float(np.max(np.abs(p_all.mean(axis=(-2, -1))))),
        "mean_kinetic_energy": float(energy.mean()),
        "mean_enstrophy": float(enstrophy.mean()),
        "mean_temporal_vorticity_std": float(omega.std(axis=0).mean()),
    }


def electro_thermal_diagnostics(
    state: np.ndarray,
    time: np.ndarray,
    config: dict[str, Any],
    result: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Diagnostics for one steady complex-field/temperature realization."""

    fields = np.asarray(state)
    if fields.shape[0] != 1 or fields.shape[1] != 3:
        raise ValueError(f"expected electro-thermal state [1,3,H,W], got {fields.shape}")
    electric = fields[0, 0] + 1j * fields[0, 1]
    temperature = fields[0, 2]

    def scalar(name: str, default: float = float("nan")) -> float:
        if result is None or name not in result:
            return default
        return float(np.asarray(result[name]))

    return {
        "finite": bool(np.isfinite(fields).all()),
        "maximum_coupling_iterations": int(scalar("coupling_iterations", 0.0)),
        "maximum_relative_temperature_update": scalar("relative_temperature_update"),
        "maximum_relative_conductivity_update": scalar("relative_conductivity_update"),
        "relative_helmholtz_residual": scalar("relative_helmholtz_residual"),
        "relative_heat_residual": scalar("relative_heat_residual"),
        "relative_robin_boundary_residual": scalar("relative_robin_boundary_residual"),
        "minimum_temperature": float(temperature.min()),
        "maximum_temperature": float(temperature.max()),
        "minimum_abs_electric_field": float(np.abs(electric).min()),
        "maximum_abs_electric_field": float(np.abs(electric).max()),
    }


def mass_transport_fluid_diagnostics(
    state: np.ndarray,
    time: np.ndarray,
    config: dict[str, Any],
    result: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Flux-based diagnostics for one Elder-type transient trajectory."""

    fields = np.asarray(state, dtype=np.float64)
    if fields.ndim != 4 or fields.shape[1] != 3:
        raise ValueError(f"expected mass-transport state [T,3,H,W], got {fields.shape}")
    ux, uy, concentration = fields[:, 0], fields[:, 1], fields[:, 2]
    density = config["rho0"] + config["density_coefficient"] * np.maximum(
        concentration, 0.0
    )
    seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
    x_count = fields.shape[-1]
    y_count = fields.shape[-2]
    dx = config["domain_length"] / (x_count - 1)
    dy = config["domain_height"] / (y_count - 1)

    def scalar(name: str, default: float = 0.0) -> float:
        if result is None or name not in result:
            return default
        return float(np.asarray(result[name]))

    mass_residuals: list[float] = []
    transport_residuals: list[float] = []
    boundary_residuals: list[float] = []
    if result is not None and {
        "velocity_x_face",
        "velocity_y_face",
        "source_field",
    }.issubset(result):
        velocity_x = np.asarray(result["velocity_x_face"], dtype=np.float64)
        velocity_y = np.asarray(result["velocity_y_face"], dtype=np.float64)
        source = np.asarray(result["source_field"], dtype=np.float64)

        def divergence(face_x, face_y):
            return (
                (face_x[:, 1:] - face_x[:, :-1]) / dx
                + (face_y[1:, :] - face_y[:-1, :]) / dy
            )

        for index in range(1, time.size - 1):
            centered_seconds = (time[index + 1] - time[index - 1]) * seconds_per_year
            temporal_mass = (
                config["porosity"]
                * (density[index + 1] - density[index - 1])
                / centered_seconds
            )
            rho_x = np.zeros_like(velocity_x[index])
            rho_y = np.zeros_like(velocity_y[index])
            rho_x[:, 1:-1] = 0.5 * (
                density[index, :, :-1] + density[index, :, 1:]
            )
            rho_y[1:-1, :] = 0.5 * (
                density[index, :-1, :] + density[index, 1:, :]
            )
            mass_x = rho_x * velocity_x[index]
            mass_y = rho_y * velocity_y[index]
            mass_divergence = divergence(mass_x, mass_y)
            mass_residuals.append(
                float(
                    np.linalg.norm(temporal_mass + mass_divergence)
                    / (
                        np.linalg.norm(temporal_mass)
                        + np.linalg.norm(mass_divergence)
                        + EPS
                    )
                )
            )

            current = concentration[index]
            species_x = np.zeros_like(velocity_x[index])
            species_y = np.zeros_like(velocity_y[index])
            species_x[:, 1:-1] = velocity_x[index, :, 1:-1] * np.where(
                velocity_x[index, :, 1:-1] >= 0.0,
                current[:, :-1],
                current[:, 1:],
            )
            species_y[1:-1, :] = velocity_y[index, 1:-1, :] * np.where(
                velocity_y[index, 1:-1, :] >= 0.0,
                current[:-1, :],
                current[1:, :],
            )
            temporal_transport = (
                config["porosity"]
                * (concentration[index + 1] - concentration[index - 1])
                / centered_seconds
            )
            advective = divergence(species_x, species_y)
            diffusion = np.zeros_like(current)
            diffusion[1:-1, 1:-1] = -config["porosity"] * config["diffusivity"] * (
                (
                    current[1:-1, 2:]
                    - 2.0 * current[1:-1, 1:-1]
                    + current[1:-1, :-2]
                )
                / dx**2
                + (
                    current[2:, 1:-1]
                    - 2.0 * current[1:-1, 1:-1]
                    + current[:-2, 1:-1]
                )
                / dy**2
            )
            residual = (
                temporal_transport + advective + diffusion - source
            )[1:-1, 1:-1]
            pieces = (
                temporal_transport[1:-1, 1:-1],
                advective[1:-1, 1:-1],
                diffusion[1:-1, 1:-1],
                source[1:-1, 1:-1],
            )
            transport_residuals.append(
                float(
                    np.linalg.norm(residual)
                    / (sum(np.linalg.norm(piece) for piece in pieces) + EPS)
                )
            )
            exterior_mass = np.concatenate(
                (mass_x[:, 0], mass_x[:, -1], mass_y[0, :], mass_y[-1, :])
            )
            exterior_species = np.concatenate(
                (species_x[:, 0], species_x[:, -1], species_y[0, :])
            )
            boundary_residuals.append(
                float(
                    (np.linalg.norm(exterior_mass) + np.linalg.norm(exterior_species))
                    / (
                        np.linalg.norm(mass_x)
                        + np.linalg.norm(mass_y)
                        + np.linalg.norm(species_x)
                        + np.linalg.norm(species_y)
                        + EPS
                    )
                )
            )

    pressure_mean = 0.0
    if result is not None and "pressure" in result:
        pressure = np.asarray(result["pressure"], dtype=np.float64)
        pressure_mean = float(np.max(np.abs(pressure.mean(axis=(-2, -1)))))
    picard_iterations = 0
    if result is not None and "picard_iterations" in result:
        values = np.asarray(result["picard_iterations"])
        picard_iterations = int(values.max(initial=0))
    compatibility_projection = 0.0
    if result is not None and "compatibility_mismatches" in result:
        mismatches = np.asarray(result["compatibility_mismatches"])
        compatibility_projection = float(np.max(np.abs(mismatches), initial=0.0))
    return {
        "finite": bool(np.isfinite(fields).all()),
        "minimum_concentration_before_tolerance_cleanup": scalar(
            "minimum_concentration_before_cleanup", float(concentration.min())
        ),
        "maximum_concentration": float(concentration.max()),
        "maximum_density": float(density.max()),
        "maximum_speed": float(np.sqrt(ux**2 + uy**2).max()),
        "relative_mass_balance_residual": float(max(mass_residuals, default=0.0)),
        "relative_transport_residual": float(max(transport_residuals, default=0.0)),
        "relative_boundary_flux_residual": float(max(boundary_residuals, default=0.0)),
        "maximum_pressure_mean": pressure_mean,
        "maximum_picard_iterations": picard_iterations,
        "reduced_timestep_count": int(scalar("reduced_timestep_count", 0.0)),
        "rejected_timestep_count": int(scalar("rejected_timestep_count", 0.0)),
        "maximum_continuity_compatibility_projection": compatibility_projection,
    }


DIAGNOSTIC_FUNCTIONS = {
    "burgers": burgers_diagnostics,
    "ks": ks_diagnostics,
    "brusselator": brusselator_diagnostics,
    "kolmogorov": kolmogorov_diagnostics,
    "electro_thermal": electro_thermal_diagnostics,
    "mass_transport_fluid": mass_transport_fluid_diagnostics,
}


def compute_diagnostics(
    case: str,
    state: np.ndarray,
    time: np.ndarray,
    config: dict[str, Any],
    result: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    function = DIAGNOSTIC_FUNCTIONS[case]
    if case in {"electro_thermal", "mass_transport_fluid"}:
        return function(state, time, config, result)
    return function(state, time, config)
